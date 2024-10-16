import numpy as np

from utils.physics.utils.grid.bidimensional_grid import BidimensionalGrid
from utils.physics.utils.grid.grid_utils import load_NetCDF, reduce_grid


class BathymetryGrid(BidimensionalGrid):
    @classmethod
    def create_from_NetCDF(Grid, NetCDF_path, lat_bounds=None, lon_bounds=None, **kwargs):
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]
        grid, lat, lon, NetCDF_data = load_NetCDF(NetCDF_path, "elevation", lat_bounds, lon_bounds)
        grid, lat, lon = reduce_grid(grid, lat, lon, 3, np.max)
        return BathymetryGrid(grid, lat, lon)
