import netCDF4 as nc
import pickle

import geopy.distance
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize
import arlpy
import arlpy.uwapm as pm
from pyproj import Geod
from scipy.optimize import least_squares
from tqdm import tqdm

from utils.physics.utils import deg_to_m, euclidian_distance

MONTHS_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# given a velocity profile, extrapolate some points using the 5 previous ones
def extrapolate_velocities_linear(depths, velocities, to_extrapolate):
    slope = (velocities[-3, :, :] - velocities[-1, :, :]) / (depths[-3] - depths[-1])
    extrapolated = velocities[-1, :, :] + slope * (to_extrapolate - depths[-1])[:, np.newaxis, np.newaxis]
    return extrapolated


# abstract class of a SoundModel, defining what a SoundModel should do
class SoundModel():
    # return the time, in s, that a sound emitted at one of the positions would require to reach the other one
    def get_sound_travel_time(self, pos1, pos2, month=None):
        return None

    def localize_common_source(self, sensors_positions, detection_times, x_min=-np.inf, y_min=-np.inf, x_max=np.inf,
                             y_max=np.inf, initial_pos=None):

        x_min, x_max = x_min or -180, x_max or 180
        y_min, y_max = y_min or -90, y_max or 90

        detection_times = np.array([(d-np.min(detection_times)).total_seconds() for d in detection_times])
        sensors_positions = np.array(sensors_positions)

        def f(x):
            time_s = x[0]
            pos = x[1:]
            array = np.zeros(len(sensors_positions))
            for i in range(len(sensors_positions)):
                try:
                    arrival = time_s + self.get_sound_travel_time(pos, sensors_positions[i])
                    array[i] = np.abs(arrival - detection_times[i])
                except:
                    array[i] = np.inf
            return array

        x0 = [0, *(initial_pos or np.mean(sensors_positions, axis=0))]
        try:
            res = least_squares(f, x0, bounds=([-np.inf, x_min, y_min], [0, x_max, y_max]))
        except:
            # absurd detection times can lead to an error during the least squares
            res = [-1, -1]


        return res

# sound model of homogeneous sound velocity
class HomogeneousSoundModel(SoundModel):
    # sound speed in m/s
    def __init__(self, sound_speed=1480):
        super().__init__()
        self.sound_speed = sound_speed

    def get_sound_travel_time(self, pos1, pos2, month=None):
        distance = geopy.distance.geodesic(pos1[:2], pos2[:2]).m
        return distance / self.sound_speed

# sound model using a monthly grid of velocities organized along lat, lon
class MonthlyGridSoundModel(SoundModel):
    # sound speed in m/s
    def __init__(self, profiles_checkpoint=None,
                 temperatures_files=None, salinities_files=None,
                 lat_bounds=None, lon_bounds=None,
                 depths=None, step_paths=0.1):
        super().__init__()
        self.step_paths = step_paths
        self.lat_bounds, self.lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]  # grid bounds
        self.velocities = []  # contains velocities profiles organized in the grid for each month

        if not profiles_checkpoint:
            assert temperatures_files and salinities_files, ("if no velocities are provided, "
                                                             "temperatures and salinities are needed to compute them")
            self.depths = np.array(depths)
            temperatures, depths, self.lat, self.lon = self.load_netcdf(temperatures_files)
            salinities, _, _, _ = self.load_netcdf(salinities_files)
            self.make_grid(temperatures, salinities, depths)
        else:
            self.load_profiles(profiles_checkpoint)
        self.velocities = np.array(self.velocities)

    def load_netcdf(self, paths):
        res = []  # list of resulting data for each month of the year
        # we expect one file per month
        assert len(paths) == 12, "we expect one profile file per month!"
        for m in tqdm(range(12), desc="loading files"):
            data = nc.Dataset(paths[m])
            resolution = np.diff(data.variables['lat'][:2])[0]
            lat_bounds = int((90 + self.lat_bounds[0]) / resolution), int((90 + self.lat_bounds[1]) / resolution)
            lon_bounds = int((180 + self.lon_bounds[0]) / resolution), int((180 + self.lon_bounds[1]) / resolution)
            var = 't_an' if 'Temperature' in data.keywords else 's_an'
            res.append(data.variables[var][0, :, lat_bounds[0]:lat_bounds[1], lon_bounds[0]:lon_bounds[1]])
            res[-1] = np.ma.filled(res[-1], fill_value=np.nan)
            res[-1][:, np.any(np.isnan(res[-1][:, :, :]), axis=0)] = np.nan  # put to nan cells where we have few pts

        depths, lat, lon = (np.array(data.variables['depth']),
                            np.array(data.variables['lat'])[lat_bounds[0]:lat_bounds[1]],
                            np.array(data.variables['lon'])[lon_bounds[0]:lon_bounds[1]])
        return res, depths, lat, lon

    def make_grid(self, temperatures, salinities, depths):
        depths_to_extrapolate = self.depths[np.argmin(np.abs(depths[-1] - self.depths)) + 1:]
        for m in tqdm(range(12), desc="computing velocities"):
            self.velocities.append(np.array([arlpy.uwa.soundspeed(temperatures[m][i, :, :], salinities[m][i, :, :],
                                                                  depths[i]) for i in range(len(depths))]))
            self.velocities[m] = np.concatenate((self.velocities[m],
                                                 extrapolate_velocities_linear(depths, self.velocities[m],
                                                                        depths_to_extrapolate)))

            mask = np.isnan(self.velocities[m])
            self.velocities[m][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.velocities[m][~mask])

    def save_profiles(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.velocities, self.depths, self.lat, self.lon), f)

    def load_profiles(self, path):
        with open(path, 'rb') as f:
            self.velocities, self.depths, self.lat, self.lon = pickle.load(f)

    def get_sound_travel_time(self, pos1, pos2, month=0):
        if not self.lat[0] < pos1[0] < self.lat[-1] or not self.lat[0] < pos2[0] < self.lat[-1] or not self.lon[0] < pos1[1] < self.lon[-1] or not self.lon[0] < pos2[1] < self.lon[-1]:
            print(f"A path between {pos1} and {pos2} is outside the velocity grid.")
            return None
        return self.get_sound_travel_time_linear(pos1, pos2, month)

    def get_sound_travel_time_linear(self, pos1, pos2, month=0):
        velocities = self.get_velocities_along_path(pos1, pos2, month)

        nans = np.any(np.isnan(velocities), axis=1)
        not_nans = ~nans
        velocities[nans] = np.mean(velocities[not_nans], axis=0)

        vel = []
        for v in velocities:
            vel.append(0.995*np.mean(v[v<=(v[0]+0.5*(np.min(v)-v[0]))]))

        return deg_to_m(self.step_paths) * np.sum(1 / np.array(vel))

    def get_sound_travel_time_bellhop(self, pos1, pos2, month=0):
        env = self.get_env(pos1, pos2, month)
        return np.mean(pm.compute_arrivals(env).time_of_arrival)

    def get_velocities_along_path(self, pos1, pos2, month=0):
        d = geopy.distance.geodesic(pos1, pos2).nautical / 60
        npoints = int(np.ceil(d / self.step_paths))
        geoid = Geod(ellps="WGS84")
        points = np.array(geoid.npts(pos1[1], pos1[0], pos2[1], pos2[0], npoints))  # caution : (lon, lat format)
        velocities, _, _ = self.get_linterpolated_profile(points[:,1], points[:,0], [month])
        return velocities[:, 0, :]
    def get_env(self, pos1, pos2, month):
        velocities = self.get_velocities_along_path(pos1, pos2, month)
        duplicated_velocities = np.concatenate((velocities, velocities))
        df = pd.DataFrame({deg_to_m(i * self.step_paths): v for i, v in enumerate(duplicated_velocities)}, index=self.depths)
        env = pm.create_env2d(depth=self.depths[-1], soundspeed=df)
        env["rx_depth"] = self.depths[np.argmin(velocities[-1])]
        env["rx_range"] = geopy.distance.geodesic(pos1[:2], pos2[:2]).m
        env["tx_depth"] = self.depths[np.argmin(velocities[0])]
        env["frequency"] = 20
        env['min_angle'] = -5
        env['max_angle'] = 5
        return env

    def get_nearest_point(self, lat, lon):
        return np.argmin(np.abs(self.lat - lat)), np.argmin(np.abs(self.lon - lon))

    def get_nearest_profile(self, lat, lon, months=None):
        months = months or range(12)
        lat, lon = self.get_nearest_point(lat, lon)
        data = self.velocities[months, :, lat, lon]
        return data, lat, lon

    def get_linterpolated_profile(self, lat, lon, months=None):
        lat, lon = np.array(lat).reshape(-1), np.array(lon).reshape(-1)
        months = months or range(12)
        latdiff = lat[:, np.newaxis]-self.lat
        londiff = lon[:, np.newaxis]-self.lon
        latdiff[latdiff < 0] = np.inf  # exclude negative values from search
        londiff[londiff < 0] = np.inf  # exclude negative values from search
        lat_i, lon_i = np.argmin(latdiff, axis=-1), np.argmin(londiff, axis=-1)
        lats_used, lons_used = [], []

        data = np.zeros((len(lat), len(months), len(self.depths)))
        threshs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        coefs = np.zeros((len(lat), 4))
        for i, (latthresh, lonthresh) in enumerate(threshs):
            lats_used.append(lat_i + latthresh)
            lons_used.append(lon_i + lonthresh)
            coefs[:,i] = [geopy.distance.geodesic([lat[j], lon[j]],
                                             [self.lat[lats_used[-1][j]], self.lon[lons_used[-1][j]]]).nautical
                      for j in range(len(lat))]
        coefs = (1 - coefs / np.array(np.sum(coefs, axis=-1))[:,np.newaxis]).reshape(-1,4) / 3
        for i, (latthresh, lonthresh) in enumerate(threshs):
            data[:] += coefs[:, i, np.newaxis, np.newaxis] * self.velocities[months, :, lat_i + latthresh, lon_i + lonthresh].reshape((len(lat), len(months), -1))

        return data, lats_used, lons_used

    def resample_profiles(self, step=200):
        res = []
        new_depths = list(range(0, 2000, step))
        new_depths = np.array(new_depths + list(range(new_depths[-1]+5*step, 5000, 5*step)))
        if len(new_depths) == len(self.depths) and np.all(new_depths == self.depths):
            return  # the new sampling is not different from the current one
        for i in new_depths:
            diff = i - self.depths
            diff[diff < 0 ] = np.inf
            d = np.argmin(diff)
            if diff[d]==0:
                res.append(self.velocities[:, d, :, :])
            else:
                slope = ((self.velocities[:, d+1, :, :] - self.velocities[:, d, :, :]) /
                         (self.depths[d+1] - self.depths[d]))
                res.append(self.velocities[:, d, :, :] + diff[d] * slope)
        self.velocities = np.moveaxis(np.array(res), 0, 1)
        self.depths = new_depths

    def localize_common_source(self, sensors_positions, detection_times, x_min=-np.inf, y_min=-np.inf, x_max=np.inf,
                             y_max=np.inf, initial_pos=None):

        x_min, x_max = x_min or -180, x_max or 180
        y_min, y_max = y_min or -90, y_max or 90

        velocities = [self.get_velocities_along_path(initial_pos, sensors_positions[i], detection_times[i].month)
                      for i in range(len(detection_times))]
        nans = [np.any(np.isnan(v), axis=(1)) for v in velocities]
        not_nans = [~n for n in nans]
        for i in range(len(velocities)):
            velocities[i][nans[i]] = np.mean(velocities[i][not_nans[i]], axis=0)
            velocities[i] = len(velocities[i]) / np.sum(1 / np.min(velocities[i], axis=1))

        detection_times = np.array([(d-np.min(detection_times)).total_seconds() for d in detection_times])
        sensors_positions = np.array(sensors_positions)




        def f(x):
            time_s = x[0]
            pos = x[1:]
            array = np.zeros(len(sensors_positions))
            for i in range(len(sensors_positions)):
                try:
                    arrival = time_s + geopy.distance.geodesic(pos, sensors_positions[i]).m / velocities[i]
                    array[i] = np.abs(arrival - detection_times[i])
                except:
                    array[i] = np.inf
            return array

        x0 = [0, *(initial_pos or np.mean(sensors_positions, axis=0))]
        try:
            res = least_squares(f, x0, bounds=([-np.inf, x_min, y_min], [0, x_max, y_max]))
        except:
            # absurd detection times can lead to an error during the least squares
            res = [-1, -1]


        return res

    def save_profile(self, lat, lon, file, months=None, linterpolated=False):
        months = months or range(12)
        if linterpolated:
            to_write, lats, lons = self.get_linterpolated_profile(lat, lon, months)
            header = (f"Celerity profiles obtained from WOA18 at ({lat},{lon}) "
                      f"(linearly interpolated from [" +
                      ",".join([f"({self.lat[lat]},{self.lon[lon]})" for lat, lon in zip(lats, lons)]) +
                      "]) - linear extrapolation used after 1500m.\n")
        else:
            to_write, lat, lon = self.get_nearest_profile(lat, lon, months)
            header = (f"Celerity profiles obtained from WOA18 at ({self.lat[lat]},{self.lon[lon]}) - linear "
                      f"extrapolation used after 1500m.\n")

        with open(file, "w") as f:
            f.write(header)
            f.write(" ".join(["{:<8}".format(t) for t in
                              ["depth"] + list(np.array(MONTHS_NAMES)[months])]) + "\n")
            for d in range(len(self.depths)):
                f.write(" ".join(["{:<8.0f}".format(self.depths[d])] + ["{:<8.2f}".format(to_write[m][d]) for m in
                                                                        range(12)]) + "\n")

        return to_write

if __name__ == '__main__':
    model = HomogeneousSoundModel()
    time_s = model.get_sound_travel_time((-12, 45), (-13, 45))
    print("Homogeneous sound travel time estimation :", time_s)

    t = [f"/home/plerolland/Bureau/profiles/temp/woa18_decav_t{i:02d}an01.csv" for i in range(1, 13)]
    s = [f"/home/plerolland/Bureau/profiles/sal/woa18_decav_s{i:02d}an01.csv" for i in range(1, 13)]
    model_grid = MonthlyGridSoundModel(t, s, [-72, 25], [0, 180])
