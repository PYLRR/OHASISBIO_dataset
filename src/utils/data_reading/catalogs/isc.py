import datetime
import re

import numpy as np

from utils.data_reading.catalogs.catalog import AcousticEvent, Emission, CatalogFile


class ISC_event(AcousticEvent, Emission):
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
        :param magnitudes: Gives a list of all computed magnitudes in the shape (author, type, value)
        """
        super().__init__(date, lat, lon, depth)
        self.ID, self.author, self.depfix, self.magnitudes = ID, author, depfix, magnitudes

    def __str__(self):
        """ Outputs a simple description of the event, specifying the event is from ISC.
        :return: A string describing the event.
        """
        return f"{self.ID} - {super().__str__()}"


class ISC_file(CatalogFile):
    """ This class represents ISC files and enables to read and access their content.
    """
    def _process_file(self, path):
        """ Read the ISC ASCII file.
        :param path: Path of the ISC ASCII file.
        :return: None
        """
        with open(path, 'rb') as file:
            lines = file.read().decode('ascii').split("\n")
        for line in lines:
            if line == "" or line[0] == "#":
                # we are still in the header or at the end of the file
                continue
            args = line.replace(" ", "").split(",")
            ID, lat, lon = int(args[0]), float(args[4]), float(args[5])
            author = args[1]
            depfix = args[7] == "TRUE"

            # put depth to nan if we don't know it
            depth = float("nan") if args[6] == "" or float(args[6]) == "0" else float(args[6])

            # if no decimal is given for the seconds, we add one for the parsing
            args[3] = f"{args[3]}.0" if "." not in args[3] else args[3]
            date = datetime.datetime.strptime(f"{args[2]}_{args[3]}", "%Y-%m-%d_%H:%M:%S.%f")

            magnitudes = []
            for i in range(8, len(args), 3):
                if args[i] == "":
                    continue
                magnitudes.append((args[i], args[i + 1], float(args[i + 2])))
            self.items[ID] = ISC_event(date, lat, lon, depth, ID, author, depfix, magnitudes)
