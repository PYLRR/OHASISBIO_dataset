import scipy

from utils.data_reading.sound_data.station import Station


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

    def get_pos(self, include_depth=False):
        """ Get an array representing the position of the event.
        :param include_depth: If true, the array will include depth.
        :return: An array giving the position of the event.
        """
        pos = [self.lat, self.lon]
        if include_depth:
            pos.append(self.depth)
        return pos

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
        :param level: The source level of the event.
        """
        super().__init__(date, lat, lon, depth)
        self.level = level

class Emission():
    """ This class represents an emission. """
    pass

class Reception():
    """ This class represents a reception. """
    pass

class AcousticEmission(AcousticEvent, Emission):
    """ This class represents generic acoustic source event. """
    def __init__(self, date, lat=None, lon=None, source_level=None):
        """ Initializes the event.
        :param date: date of the emission.
        :param lat: latitude of the emission point.
        :param lon: longitude of the emission point.
        :param source_level: source level of the emitted signal.
        """
        super().__init__(date, lat, lon)
        self.source_level = source_level

class AcousticReception(AcousticEvent, Reception):
    """ This class represents generic acoustic reception events."""
    def __init__(self, station, date, received_level=None):
        """ Initializes the event.
        :param station: station receiving the signal as a Station object.
        :param date: date of the emission.
        :param received_level: received level of the received signal.
        """
        assert isinstance(station, Station), \
            "The station provided for an AcousticReception event is not a Station object."
        super().__init__(date, station.lat, station.lon)
        self.station = station
        self.received_level = received_level

class AcousticReceptionWithData(AcousticReception):
    """ This class represents acoustic reception events for which we have data in a .wav file."""
    def __init__(self, station, date, path, received_level=None):
        """ Initializes the event without importing the data.
        :param station: station receiving the signal as a Station object.
        :param date: date of the emission.
        :param path: path of the acoustic data, centered on the event.
        :param received_level: received level of the received signal.
        """
        super().__init__(station, date, received_level)
        self.path = path
        self.sampling_f, self.data = None, None

    def get_data(self):
        """ Return data, reading them if necessary.
        :return: The acoustic data.
        """
        if self.data is None:
            self.read_data()
        return self.data

    def read_data(self):
        """ Read the data file using scipy.io ans save it as attributes.
        :return: None
        """
        self.sampling_f, self.data = scipy.io.wavfile.read(self.path)

    def forget_data(self):
        """ Forget the data file content.
        :return: None
        """
        self.data = None