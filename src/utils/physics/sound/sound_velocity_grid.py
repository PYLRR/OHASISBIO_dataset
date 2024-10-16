import geopy.distance
import numpy as np
import scipy
from scipy.optimize import least_squares

from utils.physics.sound.sound_model import SoundModel, HomogeneousSoundModel
from utils.physics.utils.grid.bidimensional_grid import BidimensionalGrid
from utils.physics.utils.grid.grid_utils import load_NetCDF


class SoundVelocityGrid(BidimensionalGrid, SoundModel):
    @classmethod
    def create_from_NetCDF(Grid, NetCDF_path, lat_bounds=None, lon_bounds=None, interpolate=False, **kwargs):
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]
        grid, lat, lon, NetCDF_data = load_NetCDF(NetCDF_path, "celerity", lat_bounds, lon_bounds)
        grid = SoundVelocityGrid(grid, lat, lon)
        grid.interpolate = interpolate
        return grid

    def get_sound_speed(self, pos1, pos2):
        if self.interpolate:
            velocities = self.get_along_path_interpolated(pos1, pos2)[0]
        else:
            velocities = self.get_along_path_nearest(pos1, pos2)[0]
        velocities = velocities[~np.isnan(velocities)]
        return scipy.stats.hmean(velocities)

    def get_sound_travel_time(self, pos1, pos2, date=None):
        distance = geopy.distance.geodesic(pos1, pos2).m
        return distance / self.get_sound_speed(pos1, pos2)

class MonthlySoundVelocityGrid(SoundModel):
    def __init__(self, paths, lat_bounds=None, lon_bounds=None, interpolate=False):
        self.models = [SoundVelocityGrid.create_from_NetCDF(p, lat_bounds, lon_bounds, interpolate) for p in paths]

    def get_sound_travel_time(self, pos1, pos2, date):
        return self.models[date.month-1].get_sound_travel_time(pos1, pos2, date)

class MonthlySoundVelocityGridOptimized(SoundModel):
    def __init__(self, paths, lat_bounds=None, lon_bounds=None, interpolate=False):
        self.models = [SoundVelocityGrid.create_from_NetCDF(p, lat_bounds, lon_bounds, interpolate) for p in paths]

    def get_sound_travel_time(self, pos1, pos2, date):
        return self.models[date.month-1].get_sound_travel_time(pos1, pos2, date)

    def localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, initial_pos=None):

        if initial_pos is None:
            initial_pos = HomogeneousSoundModel().localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max, y_max, initial_pos).x[1:]

        reference_date = detection_times[0]  # to determine the part of the year concerned
        min_date = np.argmin(detection_times)
        detection_times = np.array([(d - detection_times[min_date]).total_seconds() for d in detection_times])
        sensors_positions = np.array(sensors_positions)
        sensor_velocities = [0.996*self.models[reference_date.month-1].get_sound_speed(initial_pos, p) for p in sensors_positions]

        def f(x):
            time_s = x[0]
            pos = x[1:]
            array = np.zeros(len(sensors_positions))
            for i in range(len(sensors_positions)):
                try:
                    arrival = time_s + geopy.distance.geodesic(pos, sensors_positions[i]).m / sensor_velocities[i]
                    array[i] = np.abs(arrival - detection_times[i])
                except:
                    array[i] = np.inf
            return array

        try:
            x0 = [0, *(initial_pos)]
            x0[0] = -self.get_sound_travel_time(x0[1:], sensors_positions[min_date], reference_date)
            res = least_squares(f, x0, bounds=([-np.inf, x_min, y_min], [0, x_max, y_max]))
        except:
            # absurd detection times can lead to an error during the least squares
            res = [-1, -1]


        return res
