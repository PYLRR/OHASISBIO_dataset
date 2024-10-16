import numpy as np

from utils.physics.utils.grid.grid_utils import load_NetCDF, reduce_grid


class TridimensionalGrid():
    def __init__(self, data, lat, lon, depth):
        """ 3D physics grid class, giving values for each lat/lon/depth.
        :param data: Grid data as a 3D array.
        :param lat: Latitudes represented in the grid.
        :param lon: Longitudes represented in the grid.
        :param depth: Depths represented in the grid.
        """
        assert len(data.shape) == 3, "Data is not 3D"
        self.data, self.lat, self.lon, self.depth = data, lat, lon, depth
        self.lat_bounds, self.lon_bounds = (self.lat[0], self.lat[-1]), (self.lat[0], self.lat[-1])
        self.lat_resolution = self.lat_bounds[1] - self.lat_bounds[0]
        self.lon_resolution = self.lon_bounds[1] - self.lon_bounds[0]

    @classmethod
    def create_from_NetCDF(TridimensionalGrid, NetCDF_path, data_name, lat_bounds=None, lon_bounds=None,
                           reduction_factor=0, reduction_function=np.mean):
        """ Constructor of Grid using a NetCDF file. Loads the grid and applies the reduction factor to its data.
        Particularity of 3d grid : also loads the "depth" of the data.
        :param NetCDF_path: Path of the data, in NetCDF format.
        :param data_name: Name of the main variable in the netCDF, like "t_an" in WOA temperature.
        :param lat_bounds: Latitude bounds as a size 2 float array (°) (points outside these bounds will be forgotten).
        :param lon_bounds: Longitude bounds as a size 2 float array (°) (points outside these bounds will be forgotten).
        :param reduction_factor: The side lengths of the grid will be divided by 2^reduction_factor
        :param reduction_function: Pooling function applied on data to reduce it.
        :return: A Grid object, initialized from the NetCDF file.
        """
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]
        grid, lat, lon, NetCDF_data = load_NetCDF(NetCDF_path, data_name, lat_bounds, lon_bounds)
        depth = np.array(NetCDF_data.variables["depth"])
        grid = np.moveaxis(grid, 0, -1)  # move depth axis at the end
        grid, lat, lon = reduce_grid(grid, lat, lon, reduction_factor, reduction_function)
        grid = TridimensionalGrid(grid, lat, lon, depth)
        return grid
