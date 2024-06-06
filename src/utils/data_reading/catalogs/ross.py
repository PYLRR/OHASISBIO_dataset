import datetime

import numpy as np

from utils.data_reading.catalogs.catalog import CatalogFile
from utils.data_reading.catalogs.events import Event, AcousticEmission


class Ross_event(AcousticEmission):
    """ This class represents events taken from the Ross catalog, providing additional specific details.
    """

    def __init__(self, date, n_s, lat, lon, lat_err, lon_err, date_err, SL):
        """ Initializes the event.
        :param date: The datetime of the event happening.
        :param n_s: The number of stations used to pick this event.
        :param lat: The latitude of the event source.
        :param lon: The longitude of the event source.
        :param date_err: The error on the datetime of the event happening.
        :param lat_err: The error on the latitude of the event source.
        :param lon_err: The error on the longitude of the event source.
        :param SL: Estimated SL of the event.
        """
        super().__init__(date, lat, lon, SL)
        self.n_s, self.lat_err, self.lon_err, self.date_err = n_s, lat_err, lon_err, date_err


class Ross_file(CatalogFile):
    """ This class represents Ross events files and enables to read and access their content.
    """

    def _process_file(self, path):
        """ Read the file.
        :param path: Path of the file.
        :return: None
        """
        with open(path, 'rb') as file:
            content = file.read()
        lines = content.decode('ascii').split("\n")
        lines = [line for line in lines if line.strip()]  # remove empty lines
        lines = lines[9:]  # skip the header

        ID = 0
        for line in lines:
            date, n_s, lat, lon, lat_err, lon_err, date_err, SL = line.split()
            date = (datetime.datetime.strptime(date[:-3], '%Y%j%H%M') +
                    datetime.timedelta(milliseconds=100*int(date[-3:])))
            n_s, lat, lon, lat_err, lon_err, date_err, SL = (
                int(n_s), float(lat), float(lon), float(lat_err), float(lon_err), float(date_err), float(SL))
            self.items[ID] = Ross_event(date, n_s, lat, lon, lat_err, lon_err, date_err, SL)
            ID += 1