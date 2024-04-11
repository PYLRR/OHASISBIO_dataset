import datetime

import numpy as np
import yaml

from utils.data_reading.catalogs.catalog import Reception, Emission, CatalogFile, merge_catalogs
from utils.data_reading.catalogs.isc import AcousticEvent

def make_catalog_from_yaml_dataset(yaml_path, dataset, stations_catalog):
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.BaseLoader)[dataset]
    path = data["root_dir"]
    matchup_files = [MatchupFile(stations_catalog, f"{path}/{file}.csv") for file in data["catalogs"].keys()]
    return merge_catalogs(matchup_files)

class AcousticSource(AcousticEvent, Emission):
    def __init__(self, date, lat=None, lon=None, source_level=None):
        super().__init__(date, lat, lon)
        self.source_level = source_level

class AcousticDetection(AcousticEvent, Reception):
    def __init__(self, station, date, lat=None, lon=None, received_level=None):
        lat = lat or station.lat
        lon = lon or station.lon
        super().__init__(date, lat, lon)
        self.station = station
        self.received_level = received_level

class Matchup:
    def __init__(self, detections, label=None, source_date=None, source_lat=None, source_lon=None, source_level=None):
        self.detections = detections
        self.detections.sort()
        self.label = label
        self.estimated_source = AcousticSource(source_date, source_lat, source_lon, source_level)
        self.stations = [det.station for det in detections]

    def compute_source(self, sound_model, initial_pos=None):
        sensors_positions = np.array([d.station.get_pos() for d in self.detections])
        detection_times = np.array([d.date for d in self.detections])
        return sound_model.LocalizeCommonSource(sensors_positions, detection_times, initial_pos=initial_pos)


    def __lt__(self, other):
        return self.detections[0] < other.detections[0]


class MatchupFile(CatalogFile):
    def __init__(self, stations_catalog=None, path=None):
        self.stations_catalog = stations_catalog
        self.matchup_counter = 0
        super().__init__(path)

    def _process_file(self, path):
        with open(path, 'rb') as file:
            lines = file.read().decode('ascii').split("\n")
        parameters_of_matchups = {}
        for line in lines[1:]:
            if line == "":  # end of file
                continue
            args = line.split(",")
            name, date, received_level, matchup = tuple(args[:4])
            date = datetime.datetime.strptime(date, "%Y%m%d_%H%M%S.%f")
            received_level = float(received_level)

            if matchup not in parameters_of_matchups:
                label, source_date, source_lat, source_lon, source_level = tuple(args[4:])
                source_date = datetime.datetime.strptime(source_date, "%Y%m%d_%H%M%S.%f")
                source_lat, source_lon, source_level = float(source_lat), float(source_lon), float(source_level)
                parameters_of_matchups[matchup] = [[], label, source_date, source_lat, source_lon, source_level]

            stations = self.stations_catalog.by_names(name).by_date(date).stations
            if len(stations) == 0:
                continue
            parameters_of_matchups[matchup][0].append(AcousticDetection(stations[0], date, received_level=received_level))

        for p in parameters_of_matchups.values():
            self.items[self.matchup_counter] = Matchup(p[0], p[1], p[2], p[3], p[4], p[5])
            self.matchup_counter += 1
