import geopy
import numpy as np
from netCDF4 import Dataset
from pyproj import Geod

from utils.physics.utils.generic import get_geodesic
from utils.physics.utils.grid.grid_utils import load_NetCDF, reduce_grid


class BidimensionalGrid():
    def __init__(self, data, lat, lon):
        """ Generic physics grid class, giving values for each lat/lon.
        :param data: Grid data as a 2D/3D array.
        :param lat: Latitudes represented in the grid.
        :param lon: Longitudes represented in the grid.
        """
        self.data, self.lat, self.lon = data, lat, lon
        self.lat_bounds, self.lon_bounds = (self.lat[0], self.lat[-1]), (self.lat[0], self.lat[-1])
        self.lat_resolution = self.lat[1] - self.lat[0]
        self.lon_resolution = self.lon[1] - self.lon[0]

    @classmethod
    def create_from_NetCDF(Grid, NetCDF_path, data_name, lat_bounds=None, lon_bounds=None, reduction_factor=0,
                           reduction_function=np.mean):
        """ Constructor of Grid using a NetCDF file. Loads the grid and applies the reduction factor to its data.
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
        print(NetCDF_data)
        grid, lat, lon = reduce_grid(grid, lat, lon, reduction_factor, reduction_function)
        grid = Grid(grid, lat, lon)
        return grid

    def check_positions(self, pos):
        """ Check the provided coordinates are within the grid, raising an error if not.
        :param pos: The list of coordinates to check.
        :return: None.
        """
        for p in pos:
            assert (self.lat[0] - self.lat_resolution < p[0] < self.lat[-1] + self.lat_resolution and
                    self.lat[0] - self.lat_resolution < p[0] < self.lat[-1] + self.lat_resolution and
                    self.lon[0] - self.lon_resolution < p[1] < self.lon[-1] + self.lon_resolution and
                    self.lon[0] - self.lon_resolution < p[1] < self.lon[-1] + self.lon_resolution), \
                f"requested point is partially or totally outside the grid (given pos : {p})"

    def get_bottom_left_idx(self, coordinates):
        """ Return coordinates of the bottom-left closest point in the grid.
        :param coordinates: Coordinates of the point for which we want to look for a neighbor.
        :return:
        """
        coordinates = np.array(coordinates).reshape((-1, 2))  # ensure we have a 2D array
        lat_idx = np.searchsorted(self.lat, coordinates[:, 0], side="right") - 1
        lon_idx = np.searchsorted(self.lon, coordinates[:, 0], side="right") - 1
        # in case the value is before all available, take the first one
        lat_idx[lat_idx == -1] = 0
        lon_idx[lon_idx == -1] = 0
        return np.array(list(zip(lat_idx, lon_idx)))

    def get_nearest_idx(self, coordinates):
        """ Return coordinates of the closest point in the grid.
        :param coordinates: Coordinates of the point for which we want to look for a neighbor.
        :return:
        """
        coordinates = np.array(coordinates).reshape((-1, 2))  # ensure we have a 2D array
        lat_idx = np.searchsorted(self.lat, coordinates[:, 0])
        lon_idx = np.searchsorted(self.lon, coordinates[:, 1])

        # handle out of bounds situations
        lat_idx = np.clip(lat_idx, 1, len(self.lat) - 1)
        lon_idx = np.clip(lon_idx, 1, len(self.lon) - 1)

        # see if the closest is at right or left
        lat_left = self.lat[lat_idx - 1]
        lat_right = self.lat[lat_idx]
        lat_idx = np.where(np.abs(coordinates[:, 0] - lat_left) <= np.abs(coordinates[:, 0] - lat_right), lat_idx - 1, lat_idx)
        lon_left = self.lon[lon_idx - 1]
        lon_right = self.lon[lon_idx]
        lon_idx = np.where(np.abs(coordinates[:, 0] - lon_left) <= np.abs(coordinates[:, 0] - lon_right), lon_idx - 1,
                           lon_idx)
        return np.array(list(zip(lat_idx, lon_idx)))

    def get_nearest_values(self, coordinates):
        """ Return value of the closest point in the grid.
        :param coordinates: Coordinates of the point for which we want to look for a neighbor.
        :return:
        """
        coordinates_idx = self.get_nearest_idx(coordinates)
        return self.data[coordinates_idx[:, 0], coordinates_idx[:, 1]]

    def get_interpolated_values(self, coordinates):
        """ Return values interpolated from the 4 closest grid points.
        :param coordinates: Coordinates of the points for which we want to have the values.
        :return: The interpolated values.
        """
        nb_points = coordinates.shape[0]
        # idx of the bottom-left closest point
        closest_idx = self.get_bottom_left_idx(coordinates)
        lat_idx, lon_idx = closest_idx[:, 0], closest_idx[:, 1]
        lat_diff = coordinates[:, 0] - self.lat[lat_idx]
        lon_diff = coordinates[:, 0] - self.lon[lon_idx]

        lats_used, lons_used = [], []
        data = np.zeros(nb_points)
        offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]  # we interpolate with the 4 closest points

        weights = np.ones((nb_points, 4))  # participation of each of the 4 points in the interpolation
        for i, (lat_thresh, lon_thresh) in enumerate(offsets):
            # in case a requested point is outside the grid, we don't use two of the offsets
            if lat_thresh == 1:  # we take next latitude
                weights[lat_diff < 0, i] = 0
            if lon_thresh == 1:  # we take next longitude
                weights[lon_diff < 0, i] = 0
            if lat_thresh == 0:  # we take previous latitude
                weights[lat_idx == len(self.lat), i] = 0
            if lon_thresh == 0:  # we take previous longitude
                weights[lon_idx == len(self.lon), i] = 0

            # add the points in the used points list if it is used (i.e. non-null weight)
            lats_used.append((lat_idx + lat_thresh)[weights[:, i] > 0])
            lons_used.append((lon_idx + lon_thresh)[weights[:, i] > 0])

            try:
                weights[weights[:, i] == 1, i] = \
                    [geopy.distance.geodesic(coordinates[j],
                                             [self.lat[lats_used[-1][j]], self.lon[lons_used[-1][j]]]).nautical
                     for j in range(nb_points)]
            except:  # out of bounds, return closest value
                return self.data[lat_idx, lon_idx]
        weights = (1 - weights / np.array(np.sum(weights, axis=-1))[:, np.newaxis]) / 3
        for i, (latthresh, lonthresh) in enumerate(offsets):
            data[:] += weights[:, i] * self.data[lat_idx + latthresh, lon_idx + lonthresh]

        return data

    def get_along_path_nearest(self, pos1, pos2, step=None):
        """ Sample points following the geodesic between two points and return the closest values in the grid.
        :param pos1: First point of the wanted path.
        :param pos2: Second point of the wanted path.
        :param step: Distance, in degrees, between two sampled points in the used geodesic. Grid resolution if None.
        :return: Values that are the closest to points samples together with the actual step used.
        """
        self.check_positions([pos1, pos2])
        step = np.sqrt(np.square(self.lat_resolution)+np.square(self.lon_resolution)) if step is None else step
        points, actual_step = get_geodesic(pos1, pos2, step)
        values = self.get_nearest_values(points)
        return values, actual_step

    def get_along_path_interpolated(self, pos1, pos2, step=None):
        """ Get the values along coordinates, sampling points in the geodesic and interpolating values for each one.
        :param coordinates: Array of coordinates in (lat, lon) format.
        :param step: Distance, in degrees, between two sampled points in the used geodesic. Grid resolution if None.
        :return: The values along coordinates as a numpy array together with the actual step used.
        """
        self.check_positions([pos1, pos2])
        step = np.sqrt(np.square(self.lat_resolution)+np.square(self.lon_resolution)) if step is None else step
        points, actual_step = get_geodesic(pos1, pos2, step)
        data = self.get_interpolated_values(points)
        return data, actual_step

    def save_as_NetCDF(self, path, description, data_name):
        """ Save the grid as a new NetCDF file.
        :param path: Path where we want to save the file.
        :param description:
        :param data_name:
        :return:
        """
        root_grp = Dataset(path, 'w', format='NETCDF4')
        root_grp.description = description

        # dimensions
        root_grp.createDimension('lat', len(self.lat))
        root_grp.createDimension('lon', len(self.lon))

        lat = root_grp.createVariable('lat', 'f4', ('lat',))
        lat[:] = self.lat
        lon = root_grp.createVariable('lon', 'f4', ('lon',))
        lon[:] = self.lon
        data = root_grp.createVariable(data_name, 'f4', ("lat", "lon"))
        data[:, :] = self.data

        root_grp.close()