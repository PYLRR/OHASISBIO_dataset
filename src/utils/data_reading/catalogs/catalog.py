import datetime

import numpy as np

def merge_catalogs(catalogs):
    new = catalogs[0].__class__()
    for catalog in catalogs:
        new.items = new.items | catalog.items  # dictionnary merge
    return new

class Event:
    """ This class represents generic events, happening at one point at a precise time instant.
    """

    def __init__(self, date, lat, lon, depth=0.):
        """ Initializes the event.
        :param date: The datetime of the event happening.
        :param lat: The latitude of the event source.
        :param lon: The longitude of the event source.
        :param depth: The depth of the event source.
        """
        self.date, self.lat, self.lon, self.depth = date, lat, lon, depth

    def __str__(self):
        """ Outputs a simple description of the event.
        :return: A string describing the event.
        """
        return f"({self.lat},{self.lon},-{self.depth}) at {self.date}"

    def __lt__(self, other):
        """ Compares the time of happening of two different events, can be used to sort events lists.
        :param other: Another event.
        :return: True if the other event happens after this one.
        """
        return self.date < other.date

class AcousticEvent(Event):
    """ This class represents generic acoustic events.
    """

    def __init__(self, date, lat, lon, depth=0., level=0.):
        """ Initializes the event.
        :param date: The datetime of the event happening.
        :param lat: The latitude of the event source.
        :param lon: The longitude of the event source.
        :param depth: The depth of the event source.
        :param depth: The source level of the event.
        """
        super().__init__(date, lat, lon, depth)
        self.level = level

    def get_pos(self, include_depth=False):
        """ Get an array representing the position of the event.
        :param include_depth: If true, the array will include depth.
        :return: An array giving the position of the event.
        """
        pos = [self.lat, self.lon]
        if include_depth:
            pos.append(self.depth)
        return pos
class Emission(Event):
    """ This class represents a source event. """
    pass

class Reception(Event):
    """ This class represents a detection event. """
    pass

class CatalogFile:
    """ This class represents a generic catalog of events to enable reading and accessing their content.
    """
    def __init__(self, path=None):
        """ Initializes the file by loading it.
        :param path: Path of the file.
        """
        self.path = path
        self.items = {}
        if path:
            self._process_file(path)

    def get_items_list(self, sorted=True):
        """ Return the list of the items, possibly sorted.
        :param sorted: If True, sorts the items by date.
        :return: The list of items.
        """
        items = list(self.items.values())
        if sorted:
            items.sort()
        return items

    def find_nearest_items(self, date, thresh=60):
        """ Find and return the nearest item in the catalog.
        :param date: The date near which we want to find an item.
        :param thresh: The time difference, in seconds, that we allow between the found item and date.
        :return: An item if the time difference is less than thresh, else None.
        """
        diff = np.abs(np.array([e.date for e in self.items.values()])-date)
        idx = np.argmin(diff)
        if diff[idx] < datetime.timedelta(seconds=thresh):
            return list(self.items.values())[idx]
        return None

    def _process_file(self, path):
        """ Read the file.
        :param path: Path of the ISC ASCII file.
        :return: None
        """
        return None

    def __getitem__(self, ID):
        """ Enables to obtain an item by its ID, using the [] operator on an object
        :param ID: The ID of the ISC item we look for.
        :return: The item if it exists.
        """
        return self.items[ID]
