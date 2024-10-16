import netCDF4 as nc
import numpy as np
import skimage


def load_NetCDF(NetCDF_path, data_name, lat_bounds, lon_bounds, lat_name="lat", lon_name="lon"):
    """ Function to load a NetCDF file.
    :param NetCDF_path: Path of the data, in NetCDF format.
    :param data_name: Name of the main variable in the netCDF, like "t_an" in WOA temperature.
    :param lat_bounds: Latitude bounds as a size 2 float array (°) (points outside these bounds will be ignored).
    :param lon_bounds: Longitude bounds as a size 2 float array (°) (points outside these bounds will be ignored).
    :param lat_name: The name used to represent the available latitudes in the NetCDF.
    :param lon_name: The name used to represent the available longitudes in the NetCDF.
    :return: The data, lat, lon, lat resolution and lon resolution together with the initial NetCDF object.
    """

    data = nc.Dataset(NetCDF_path)

    lat_resolution = data.variables[lat_name][1] - data.variables[lat_name][0]
    lon_resolution = data.variables[lon_name][1] - data.variables[lon_name][0]

    # index bounds in the lat/lon grid given the lat/lon bounds and the resolution
    lat_bounds = (max(0,int((lat_bounds[0] - data.variables[lat_name][0]) / lat_resolution)),
                  max(0,int((lat_bounds[1] - data.variables[lat_name][0]) / lat_resolution)))
    lon_bounds = (max(0,int((lon_bounds[0] - data.variables[lon_name][0]) / lon_resolution)),
                  max(0,int((lon_bounds[1] - data.variables[lon_name][0]) / lon_resolution)))

    lat, lon = (np.array(data.variables[lat_name])[lat_bounds[0]:lat_bounds[1]],
                np.array(data.variables[lon_name])[lon_bounds[0]:lon_bounds[1]])

    # load the right variable, keeping nan values as nan in the numpy array
    grid = np.array(data.variables[data_name][lat_bounds[0]:lat_bounds[1], lon_bounds[0]:lon_bounds[1]]
                    .filled(fill_value=np.nan))
    if len(grid.shape) == 3:  # particular case of unneeded temporal axis
        grid = grid[0]

    return grid, lat, lon, data


def reduce_grid(grid, lat, lon, reduction_factor, reduction_function):
    """ Apply the reduction factor to the grid data to reduce its dimension.
    :param grid: Grid data as a 2D/3D array.
    :param lat: Latitudes represented in the grid.
    :param lon: Longitudes represented in the grid.
    :param reduction_factor: The side lengths of the grid will be divided by 2^reduction_factor
    :param reduction_function: Pooling function applied on data to reduce it.
    :return: The grid data, but reduced, along with the new lat/lon and lat/lon resolutions.
    """
    # we add a dummy third dimension in case the grid is 3D
    block = (2 ** reduction_factor, 2 ** reduction_factor) if len(grid.shape) == 2 else \
        (2 ** reduction_factor, 2 ** reduction_factor, 1)
    data = skimage.measure.block_reduce(grid, block, reduction_function)
    lat = skimage.measure.block_reduce(lat, 2 ** reduction_factor, np.mean)
    lon = skimage.measure.block_reduce(lon, 2 ** reduction_factor, np.mean)
    return data, lat, lon



