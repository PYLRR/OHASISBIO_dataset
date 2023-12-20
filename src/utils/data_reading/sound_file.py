import datetime
import wave

import scipy
import numpy as np
import math
import re
import locale

class SoundFile:
    """ Class representing a single sound file.
    """
    EXTENSION = ""  # extension of the files like ".wav"

    def __init__(self, path, skip_data=False, identifier=None):
        """ Constructor reading file metadata and content if required.
        :param path: The path of the file.
        :param skip_data: If True, we only read the metadata of the file. Else, we also read its content.
        :param identifier: The ID of this file, that will be used to compare it with another file. If unspecified,
        path is used.
        """
        self.header = {}  # dict that contains the metadata of the file
        self.data = []  # data of the file
        self.path = path
        self.identifier = self.path if identifier is None else identifier
        self._read_header()  # read the metadata of the file
        if not skip_data:
            self.read_data()  # read the data of the file

    def _read_header(self):
        """ Read the metadata of the file and update self.header.
        :return: None.
        """
        pass  # abstract method

    def read_data(self):
        """ Read the data of the file and update self.data. Public method because in case the file was previously loaded
        but without reading its data, it may be necessary to ask to read the data.
        :return: None.
        """
        pass  # abstract method

    def get_data(self, start=None, end=None):
        """ Given a start datetime and an end datetime, read the data. In case the bounds are outside of the file, the
        method does not throw an exception and simply return the data from its start, and/or to its end.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: The required data.
        """
        data = self.data
        if start is not None and end is not None:
            assert end > start, "required end is before or equal to the start"

        if start is not None:
            # select the data starting after start
            offset = start - self.header["start_date"]
            offset_points = int(offset.total_seconds() * self.header["sampling_frequency"])
            if offset_points > 0:
                data = data[offset_points:]

        if end is not None:
            # select the data ending before end
            offset = self.header["end_date"] - end
            offset_points = int(offset.total_seconds() * self.header["sampling_frequency"])
            if offset_points > 0:
                data = data[:-offset_points]
        return data

    def __eq__(self, other):
        """ Test if another file has a similar identifier, or if an item is the file identifier.
        :param other: Another SoundFile or an object of the class of self.identifier.
        :return: True if other is a SoundFile with the same identifier or if other is our identifier, else False.
        """
        if type(other) == type(self.identifier):  # the other object may be the identifier
            return self.identifier == other
        if type(other) == type(self):  # the other object is a SoundFile that may have the same identifier
            return self.identifier == other.identifier
        return False

class WavFile(SoundFile):
    """ Class representing .wav files. We expect wav files to be named with their start time as YYYYMMDD_hhmmss.
    """
    EXTENSION = "wav"

    def _read_header(self):
        """ Read the metadata of the file using its name and header and update self.header.
        :return: None.
        """
        with wave.open(self.path, 'rb') as file:
            # get information from the file header
            self.header["sampling_frequency"] = file.getframerate()
            self.header["samples"] = file.getnframes()

        duration_micro = 10 ** 6 * self.header["samples"] / self.header["sampling_frequency"]
        self.header["duration"] = datetime.timedelta(microseconds=duration_micro)
        file_name = self.path.split("/")[-1][:-4]  # get the name of the file and get rid of extension
        self.header["start_date"] = datetime.datetime.strptime(file_name, "%Y%m%d_%H%M%S")
        self.header["end_date"] = self.header["start_date"] + self.header["duration"]

    def read_data(self):
        """ Read the data of the file using scipy and update self.data.
        :return: None.
        """
        _, self.data = scipy.io.wavfile.read(self.path)
        if self.data.dtype == np.int16:  # 16 bits format : it has been divided by 2^14 and we thus put it back in uPa
            self.data = self.data.astype(np.int32)
            self.data = self.data * 2**14