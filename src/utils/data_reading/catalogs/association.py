import numpy as np

from utils.data_reading.catalogs.events import AcousticReception


class Association:
    """ Class representing the association of several acoustic receptions that are supposed to have a common source.
    """
    def __init__(self, events=None):
        """ Initializes the Association instance.
        :param detections: List of AcousticReception instances.
        """
        self.events = []
        if events is not None:
            for event in events:
                self.add_event(event)
        self.idx = -1  # to enable iteration

    def add_event(self, event):
        """ Add an AcousticReception instance to the list of events.
        :param event: AcousticReception instance.
        :return: None
        """
        assert isinstance(event, AcousticReception), "The provided event was not instance of AcousticReception."
        self.events.append(event)

    def compute_source(self, sound_model, initial_pos=None):
        """ Given a sound model and an initial position, tries to locate the source of the events of the Association.
        :param sound_model: SoundModel instance used to perform the location.
        :param initial_pos: The initial position of the location as a 2D list or numpy array in format (lat, lon).
        :return: The result of the location attempt.
        """
        sensors_positions = np.array([d.station.get_pos() for d in self.events])
        detection_times = np.array([d.date for d in self.events])
        return sound_model.localize_common_source(sensors_positions, detection_times, initial_pos=initial_pos)

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
        if self.idx < len(self.events):
            return self.events[self.idx]
        self.idx = - 1
        raise StopIteration

    def __getitem__(self, idx):
        """ Enables to obtain an item by its index, using the [] operator.
        :param idx: The idx of the event to return.
        :return: The event if it exists.
        """
        return self.events[idx]