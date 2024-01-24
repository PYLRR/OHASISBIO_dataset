import netCDF4 as nc
import pickle

import geopy.distance
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize
import arlpy
import arlpy.uwapm as pm
from tqdm import tqdm

from utils.physics.utils import deg_to_m, euclidian_distance

MONTHS_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def munk_profile(depth, epsilon, s_mid_depth, s_high_depth):
    zt = 2 * (depth - s_mid_depth) / s_high_depth
    return s_high_depth * (1 + epsilon * (zt - 1 + np.exp(-zt)))


def munk_profile_residuals(params, depths, velocities):
    epsilon, s_mid_depth, s_width = params['epsilon'], params['s_mid_depth'], params['s_width']
    s_high_depth = s_mid_depth + s_width
    res = munk_profile(depths, epsilon, s_mid_depth, s_high_depth)
    return np.abs(velocities - res)


# given a velocity profile, extrapolate a more complete and regular velocity profile
def extrapolate_velocities_munk(depths, velocities, depths_of_profiles):
    depths, velocities = np.array(depths), np.array(velocities)
    params = Parameters()
    params.add('epsilon', value=0.00737, vary=True, min=0.001, max=0.01)
    params.add('s_mid_depth', value=1300, vary=True, min=0, max=depths_of_profiles[-1])
    params.add('s_width', value=200, vary=True, min=50, max=500)

    out = minimize(munk_profile_residuals, params, args=(depths, velocities))
    s_depth = out.params['s_mid_depth'].value
    v = munk_profile(depths_of_profiles, out.params['epsilon'].value, s_depth, s_depth + out.params['s_width'].value)
    return v


# given a velocity profile, extrapolate some points using the 5 previous ones
def extrapolate_velocities(depths, velocities, to_extrapolate):
    slope = (velocities[-3, :, :] - velocities[-1, :, :]) / (depths[-3] - depths[-1])
    extrapolated = velocities[-1, :, :] + np.array([slope * (to_extrapolate - depths[-1])[i]
                                                    for i in range(len(to_extrapolate))])
    return extrapolated


# abstract class of a SoundModel, defining what a SoundModel should do
class SoundModel():
    # return the time, in s, that a sound emitted at one of the positions would require to reach the other one
    def get_sound_travel_time(self, pos1, pos2):
        return None


# sound model of homogeneous sound velocity
class HomogeneousSoundModel(SoundModel):
    # sound speed in m/s
    def __init__(self, sound_speed=1480):
        super().__init__()
        self.sound_speed = sound_speed

    def get_sound_travel_time(self, pos1, pos2):
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
                                                 extrapolate_velocities(depths, self.velocities[m],
                                                                        depths_to_extrapolate)))

    def save_profiles(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.velocities, self.depths, self.lat, self.lon), f)

    def load_profiles(self, path):
        with open(path, 'rb') as f:
            self.velocities, self.depths, self.lat, self.lon = pickle.load(f)

    def get_sound_travel_time(self, pos1, pos2, month=0):
        env = self.get_env(pos1, pos2, month)
        return None

    def get_env(self, pos1, pos2, month):
        pos1, pos2 = np.array(pos1), np.array(pos2)
        resolution = np.diff(self.lat[:2])[0]
        steps = int(euclidian_distance(pos1, pos2) / self.step_paths)
        velocities = []
        for i in range(steps):
            point = pos1 + (pos2 - pos1) / steps
            closest = np.round((point - np.array([self.lat[0], self.lon[0]])) / resolution).astype(np.uint32)
            velocities.append(self.velocities[month][:, closest[0], closest[1]])
        df = pd.DataFrame({deg_to_m(i * self.step_paths): velocities[i] for i in range(steps)}, index=self.depths)
        env = pm.create_env2d(depth=self.depths[-1] if len(pos1) < 3 else pos1[2], soundspeed=df)
        return env

    def get_nearest_profile(self, lat, lon, months=None):
        months = months or range(12)
        lat, lon = np.argmin(np.abs(self.lat - lat)), np.argmin(np.abs(self.lon - lon))
        data = self.velocities[months, :, lat, lon]
        return data, lat, lon

    def get_linterpolated_profile(self, lat, lon, months=None):
        months = months or range(12)
        latdiff, londiff = lat - self.lat, lon - self.lon
        latdiff[latdiff < 0] = np.inf  # exclude negative values from search
        londiff[londiff < 0] = np.inf  # exclude negative values from search
        lat, lon = np.argmin(latdiff), np.argmin(londiff)
        lats, lons = [], []

        data = np.zeros((len(months), len(self.depths)))
        threshs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        coefs = np.zeros(4)
        for i, (lathtresh, lonhtresh) in enumerate(threshs):
            lats.append(lat + lathtresh)
            lons.append(lon + lonhtresh)
            coefs[i] = euclidian_distance([lat, lon], [self.lat[lats[-1]], self.lon[lons[-1]]])
        coefs = (1 - coefs / np.sum(coefs))/3
        for i, (lathtresh, lonhtresh) in enumerate(threshs):
            data += coefs[i] * self.velocities[months, :, lat + lathtresh, lon + lonhtresh]

        return data, lats, lons

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
