import datetime

import numpy as np
import yaml

from utils.data_reading.sound_data.sound_file_manager import make_manager

class Station:
    def __init__(self, path, name=None, lat=None, lon=None, date_start=None, date_end=None, dataset=None,
                 initialize_metadata=False):
        self.manager = None
        self.path = path
        assert isinstance(path, str)
        assert not name or isinstance(name, str)
        assert not date_start or isinstance(date_start, datetime.datetime)
        assert not date_end or isinstance(date_end, datetime.datetime)
        assert not lat or isinstance(lat, float) or isinstance(lat, int)
        assert not lon or isinstance(lon, float) or isinstance(lon, int)
        assert not dataset or isinstance(dataset, str)
        self.name = name
        self.date_start = date_start
        self.date_end = date_end
        self.lat = lat
        self.lon = lon
        self.dataset = dataset
        if initialize_metadata:
            self.get_manager()
            self.name = name or self.manager.station_name
            self.date_start = date_start or self.manager.dataset_start
            self.date_end = date_end or self.manager.dataset_end

    def get_manager(self):
        self.load_data()
        return self.manager

    def load_data(self):
        if self.path and not self.manager:
            self.manager = make_manager(self.path)

    def get_pos(self):
        """ Get an array representing the position of the station.
        :return: An array giving the position of the station.
        """
        return [self.lat, self.lon]

    def light_copy(self):
        """ Make a copy of this station, only including metadata.
        :return: A copy of self containing the name, lat, lon, dates, path and dataset of the station.
        """
        return Station(self.path, self.name, self.lat, self.lon, self.date_start, self.date_end, self.dataset)

    def __str__(self):
        return f"station {self.name}"

    def __eq__(self, other):
        return (self.name == other.name and self.lat == other.lat and self.lon == other.lon and
                self.date_start == other.date_start and self.date_end == other.date_end)

class StationsCatalog():
    def __init__(self, yaml_file=None):
        self.stations = []

        if yaml_file:
            self.load_yaml(yaml_file)

    def load_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.load(f, Loader=yaml.BaseLoader)

        for dataset, dataset_yaml in data.items():
            path = dataset_yaml["root_dir"]
            for station_name, station_yaml in dataset_yaml["stations"].items():
                date_start, date_end, lat, lon = None, None, None, None
                if station_yaml["date_start"].strip() != "":
                    date_start = datetime.datetime.strptime(station_yaml["date_start"], "%Y%m%d_%H%M%S")
                if station_yaml["date_end"].strip() != "":
                    date_end = datetime.datetime.strptime(station_yaml["date_end"], "%Y%m%d_%H%M%S")
                if station_yaml["lat"].strip() != "":
                    lat, lon = float(station_yaml["lat"]), float(station_yaml["lon"])

                st = Station(f"{path}/{station_name}", station_name, lat, lon, date_start, date_end, dataset)
                self.stations.append(st)

    def add_station(self, station):
        self.stations.append(station)

    def load_stations(self):
        for st in self.stations:
            st.load_data()

    def by_datasets(self, datasets):
        datasets = [datasets] if type(datasets) == str else datasets
        res = StationsCatalog()
        for st in self.stations:
            if st.dataset in datasets:
                res.add_station(st)
        return res

    def by_names(self, names):
        names = [names] if type(names) == str else names
        res = StationsCatalog()
        for st in self.stations:
            if st.name in names:
                res.add_station(st)
        return res

    def by_date(self, date):
        res = StationsCatalog()
        for st in self.stations:
            if st.date_start and st.date_end and st.date_start < date < st.date_end:
                res.add_station(st)
        return res

    def by_dates_or(self, dates):
        res = StationsCatalog()
        for st in self.stations:
            for date in dates:
                if st.date_start < date < st.date_end:
                    res.add_station(st)
                    break
        return res

    def starts_before(self, date):
        res = StationsCatalog()
        for st in self.stations:
            if st.date_start < date:
                res.add_station(st)
        return res

    def ends_after(self, date):
        res = StationsCatalog()
        for st in self.stations:
            if date < st.date_end:
                res.add_station(st)
        return res

    def by_date_propagation(self, event, sound_model, delta=None):
        res = []
        times_of_prop = []
        delta = delta or datetime.timedelta(seconds=0)
        for st in self.stations:
            time_of_prop = sound_model.get_sound_travel_time(event.get_pos(), st.get_pos())
            if not time_of_prop:
                continue
            time_of_arrival = event.date + datetime.timedelta(seconds=time_of_prop)
            if st.date_start < time_of_arrival - delta < st.date_end and \
            st.date_start < time_of_arrival + delta < st.date_end:
                times_of_prop.append(time_of_prop)
                res.append((st, time_of_arrival))
        res = np.array(res)[np.argsort(times_of_prop)]
        return res

    def by_loc(self, min_lat, min_lon, max_lat, max_lon):
        res = StationsCatalog()
        for st in self.stations:
            if min_lat < st.lat < max_lat and min_lon < st.lon < max_lon:
                res.add_station(st)
        return res

    def filter_out_unlocated(self):
        res = StationsCatalog()
        for st in self.stations:
            if st.lat is not None:
                res.add_station(st)
        return res

    def filter_out_undated(self):
        res = StationsCatalog()
        for st in self.stations:
            if st.date_start is not None:
                res.add_station(st)
        return res

    def __getitem__(self, number):
        return self.stations[number]

    def __len__(self):
        return len(self.stations)