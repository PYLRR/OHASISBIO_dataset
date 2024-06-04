import datetime

import numpy as np

def merge_catalogs(catalogs):
    """ Take a list of catalogs and merge them into a single catalog.
    :param catalogs: List of catalogs to merge.
    :return: A new catalog of the class of the first one containing the merged items of all provided catalogs.
    """
    new = catalogs[0].__class__()
    for catalog in catalogs:
        new.items = new.items | catalog.items  # dictionnary merge
    return new

class Catalog:
    """ This class represents a generic catalog of events """
    def __init__(self):
        """ Initializes the catalog.
        """
        self.items = {}
        self.idx = -1  # to enable iteration

    def get_items_list(self, sorted=True):
        """ Return the list of the items, possibly sorted.
        :param sorted: If True, sorts the items.
        :return: The list of items.
        """
        items = list(self.items.values())
        if sorted:
            items.sort()
        return items

    def find_nearest_items(self, date, thresh_s=60):
        """ Find and return the nearest item in the catalog.
        :param date: The date near which we want to find an item.
        :param thresh_s: The time difference, in seconds, that we allow between the found item and date.
        :return: An item if the time difference is less than thresh, else None.
        """
        diff = np.abs(np.array([e.date for e in self.items.values()])-date)
        idx = np.argmin(diff)
        if diff[idx] < datetime.timedelta(seconds=thresh_s):
            return list(self.items.values())[idx]
        return None

    def __getitem__(self, ID):
        """ Enables to obtain an item by its ID, using the [] operator.
        :param ID: The ID of the ISC item we look for.
        :return: The item if it exists.
        """
        return self.items[ID]

    def __iter__(self):
        """ __iter__ definition to allow to iterate threw this class.
        :return: self.
        """
        return self

    def __next__(self):
        """ __next__ definition to allow to iterate threw this class.
        :return: None.
        """
        self.idx += 1
        if self.idx < len((l := list(self.items.items()))):
            return l[self.idx]
        raise StopIteration

class CatalogFile(Catalog):
    """ This class represents a generic catalog of events stored on a file. """
    def __init__(self, path=None):
        """ Initializes the file by loading it.
        :param path: Path of the file.
        """
        super().__init__()
        self.path = path
        if path:
            self._process_file(path)

    def _process_file(self, path):
        """ Read the file.
        :param path: Path of the ISC ASCII file.
        :return: None
        """
        return None
