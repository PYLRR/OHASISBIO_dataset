import datetime

import numpy as np
import obspy
from obspy import UTCDateTime, Catalog
from obspy.core.event import ResourceIdentifier, CreationInfo, Amplitude, Origin, Magnitude
from obspy.core.event.header import EventType

from utils.data_reading.catalogs.catalog import CatalogFile
from utils.data_reading.catalogs.events import Event


class ISC_event(Event):
    """ This class represents events taken from the ISC catalog, providing additional specific details.
    """

    def __init__(self, date, lat, lon, depth, ID, author, depfix, magnitudes):
        """ Initializes the event.
        :param date: The datetime of the event happening.
        :param lat: The latitude of the event source.
        :param lon: The longitude of the event source.
        :param depth: The depth of the event source (from the ground).
        :param ID: The ISC ID of the event.
        :param author: The author of the event discovery.
        :param depfix: Indicates whether a depfix has been performed or not.
        :param magnitudes: List of all computed magnitudes in the shape (author, type, value).
        """
        super().__init__(date, lat, lon, depth)
        self.ID, self.author, self.depfix, self.magnitudes = ID, author, depfix, magnitudes

    def get_magnitude(self, wanted_mag):
        """ Returns the magnitude of the given magnitude type.
        :param wanted_mag: Wanted magnitude (such as "mb", "mw"...).
        :return: An array of the wanted magnitudes, possibly empty.
        """
        wanted_mag = np.char.lower(wanted_mag)
        available_mags = np.array(self.magnitudes)
        if len(available_mags) > 0 and np.any(wanted_mag == np.char.lower(np.array(available_mags)[:, 1])):
            mag = available_mags[np.char.lower(available_mags[:, 1]) == wanted_mag]
            return mag[:, 2].astype(np.float32)
        return []

    def to_obspy(self):
        """ Converts to the relevant obspy object.
        :return: An ObsPy Event object.
        """
        return obspy.core.event.event.Event(resource_id=ResourceIdentifier(str(self.ID)),
                                                   event_type=EventType("earthquake"),
                                                   creation_info=CreationInfo(author=self.author),
                                                   origin=Origin(UTCDateTime(self.date), latitude=self.lat, longiude=self.lon),
                                                   magnitudes=[Magnitude(mag=m, magnitude_type=t) for _, t, m in
                                                               self.magnitudes])

    def __str__(self):
        """ Outputs a simple description of the event, specifying the event is from ISC.
        :return: A string describing the event.
        """
        return f"ISC event {self.ID} - {super().__str__()}"


class ISC_file(CatalogFile):
    """ This class represents ISC files and enables to read and access their content.
    """

    def _process_file(self, path):
        """ Read the ISC ASCII file.
        :param path: Path of the ISC ASCII file.
        :return: None
        """
        with open(path, 'rb') as file:
            content = file.read()
        try:
            lines = content.decode('ascii').split("\n")
        except:
            lines = content.decode('utf8').split("\n")

        idx_ID, idx_auth, idx_date, idx_time, idx_lat, idx_lon, idx_depth, idx_depfix, idx_mags = tuple([None]*9)
        for line in lines:
            if line == "" or line[0] == "#" or line[:8] == "adminisc":
                # we are still in the header or at the end of the file
                # in case we reached column names line, we look for some indexes
                if "EVENTID" in line:
                    cols = line[1:].replace(" ", "").split(",")
                    idx_ID, idx_auth, idx_date, idx_time, idx_lat, idx_lon, idx_depth, idx_depfix, idx_mags = (
                        cols.index("EVENTID"), cols.index("AUTHOR"), cols.index("DATE"), cols.index("TIME"),
                        cols.index("LAT"), cols.index("LON"), cols.index("DEPTH"), cols.index("DEPFIX"),
                        cols.index("MAG"))
                    idx_mags -= 2  # magnitudes are given as AUTHOR, TYPE, MAG and we want to start at AUTHOR
                continue
            args = line.replace(" ", "").split(",")
            ID, lat, lon = int(args[idx_ID]), float(args[idx_lat]), float(args[idx_lon])
            author = args[idx_auth]
            depfix = args[idx_depfix] == "TRUE"

            # put depth to nan if we don't know it
            depth = float("nan") if args[idx_depth] == "" or float(args[idx_depth]) == "0" else float(args[idx_depth])

            # if no decimal is given for the seconds, we add one for the parsing
            args[idx_time] = f"{args[idx_time]}.0" if "." not in args[idx_time] else args[idx_time]
            date = datetime.datetime.strptime(f"{args[idx_date]}_{args[idx_time]}", "%Y-%m-%d_%H:%M:%S.%f")

            magnitudes = []
            for i in range(idx_mags, len(args), 3):
                if args[i] == "":
                    continue
                magnitudes.append((args[i], args[i + 1], float(args[i + 2])))
            self.items[ID] = ISC_event(date, lat, lon, depth, ID, author, depfix, magnitudes)

    def to_obspy(self):
        """ Converts to the relevant obspy object.
        :return: An ObsPy Catalog object.
        """
        return Catalog([event.to_obspy() for event in self.items.values()], description="ISC catalog")