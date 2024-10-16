import datetime

import glob2

from utils.data_reading.catalogs.association import Association
from utils.data_reading.catalogs.catalog import Catalog, merge_catalogs
from utils.data_reading.catalogs.events import AcousticReceptionWithData
from utils.data_reading.catalogs.ISC import ISC_file

class TwinAssociation():
    """ Class representing a twin association, that is an acoustic Association instance linked with an ISC event."""
    def __init__(self, acoustic_association, isc_event):
        """ Initializes the TwinAssociation object.
        :param acoustic_association: Association instance.
        :param isc_event: ISC event.
        """
        self.acoustic_association = acoustic_association
        self.isc_event = isc_event


class TwinCatalog(Catalog):
    """ Class representing a twin catalog, containing items both seismic and acoustic
    """
    def __init__(self, acoustic_files_path, isc_files_path, stations):
        """ Initializes the Twin Catalog.
        :param acoustic_files_path: Path where we can find the acoustic catalog organized as
        YEAR/ISC_ID/acoustic_events.wav
        :param isc_files_path: Path where we can find the seismic catalogs organized in txt files.
        :param stations: Catalog of stations.
        """

        super().__init__()

        # ISC import
        isc_files = glob2.glob(f"{isc_files_path}/*.txt")
        isc = []
        for isc_file in isc_files:
            isc.append(ISC_file(isc_file))
        self.isc = merge_catalogs(isc)

        # acoustic import
        acoustic_events = glob2.glob(f"{acoustic_files_path}/*/*")
        for event in acoustic_events:
            ID = int(event.split("/")[-1])
            stations_used = glob2.glob(f"{event}/*")

            # make an association of all acoustic events
            association = []
            for station in stations_used:
                station_params = station.split("/")[-1][:-4].split("_")
                if "COLMEIA" in station_params[0]:
                    # particularity : colmeia stations have "_" in their names
                    station_params[0] = station_params[0]+"_"+station_params[1]
                    station_params[1:] = station_params[2:]
                station_name, date = station_params[0], datetime.datetime.strptime(
                    f"{station_params[1]}_{station_params[2]}", "%Y%m%d_%H%M%S")
                station_object = stations.by_date(date).by_names(station_name)
                assert len(station_object) > 0, f"0 station found for {station_name} - {date}"
                for st in station_object:
                    association.append(AcousticReceptionWithData(st, date, station))

            assert ID not in self.items, f"ID {ID} already in catalog."
            self.items[ID] = TwinAssociation(Association(association), self.isc[ID])