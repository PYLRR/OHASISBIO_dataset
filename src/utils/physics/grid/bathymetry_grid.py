import geopy.distance
import netCDF4 as nc
import numpy as np
from line_profiler_pycharm import profile
from matplotlib import pyplot as plt
import skimage.measure
from pyproj import Geod


class BathymetryModel():
    def __init__(self, GEBCO_ncdf_path, lat_bounds=None, lon_bounds=None, step_paths=0.05, reduction_factor=3):
        self.step_paths = step_paths
        self.lat_bounds, self.lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]  # grid bounds
        self.data, self.lat, self.lon, self.resolution = self.load_grid(GEBCO_ncdf_path, reduction_factor)

    def load_grid(self, GEBCO_ncdf_path, reduction_factor):
        data = nc.Dataset(GEBCO_ncdf_path)
        resolution = np.diff(data.variables['lat'][:2])[0]
        lat_bounds = int((90 + self.lat_bounds[0]) / resolution), int((90 + self.lat_bounds[1]) / resolution)
        lon_bounds = int((180 + self.lon_bounds[0]) / resolution), int((180 + self.lon_bounds[1]) / resolution)

        res = np.array(data.variables["elevation"][lat_bounds[0]:lat_bounds[1], lon_bounds[0]:lon_bounds[1]])
        res[res>0] = 0  # no need to keep elevations
        lat, lon = (np.array(data.variables['lat'])[lat_bounds[0]:lat_bounds[1]],
                        np.array(data.variables['lon'])[lon_bounds[0]:lon_bounds[1]])

        res = skimage.measure.block_reduce(res, (2**reduction_factor,2**reduction_factor), np.max)
        lat = skimage.measure.block_reduce(lat, 2**reduction_factor, np.mean)
        lon = skimage.measure.block_reduce(lon, 2**reduction_factor, np.mean)

        return res, lat, lon, resolution * 2**reduction_factor

    def get_bathymetry_along_path(self, pos1, pos2):
        assert (self.lat[0]-self.resolution < pos1[0] < self.lat[-1]+self.resolution and
                self.lat[0]-self.resolution < pos2[0] < self.lat[-1]+self.resolution and
                self.lon[0]-self.resolution < pos1[1] < self.lon[-1]+self.resolution and
                self.lon[0]-self.resolution < pos1[1] < self.lon[-1]+self.resolution), \
            f"requested profile is partially or totally outside the grid (given pos : {pos1} and {pos2})"
        d = geopy.distance.geodesic(pos1, pos2).nautical / 60
        npoints = int(np.ceil(d / self.step_paths))
        geoid = Geod(ellps="WGS84")
        points = np.array(geoid.npts(pos1[1], pos1[0], pos2[1], pos2[0], npoints))  # caution : (lon, lat format)
        bathymetries = self.get_profile(points[:, 1], points[:, 0])
        return bathymetries

    def get_profile(self, lat_series, lon_series):
        lat, lon = np.array(lat_series).reshape(-1), np.array(lon_series).reshape(-1)
        latdiff = np.abs(lat[:, np.newaxis]-self.lat)
        londiff = np.abs(lon[:, np.newaxis]-self.lon)

        lat_i, lon_i = np.argmin(latdiff, axis=-1), np.argmin(londiff, axis=-1)
        data = self.data[lat_i, lon_i]

        return data

if __name__=="__main__":
    @profile
    def main():
        bathy_model = BathymetryModel("../../../../data/geo/GEBCO_2023_sub_ice_topo.nc",
                                      lat_bounds=[-75, 25], lon_bounds=[0, 150])
        #plt.imshow(bathy_model.data[::-1,:], cmap="viridis")
        #plt.show()
        profile = bathy_model.get_bathymetry_along_path((-40, 30),(-20, 40))
        plt.plot(profile)
        plt.show()

    main()