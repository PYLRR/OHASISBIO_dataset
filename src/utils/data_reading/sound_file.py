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

class DatFile(SoundFile):
    EXTENSION = "DAT"
    TO_VOLT = 5.0 / 2 ** 24
    SENSIBILITY = -163.5

    def _read_header(self):
        with open(self.path, 'rb') as file:
            file_header = file.read(400)

        file_header = file_header.decode('ascii').split("\n")

        self.header["site"] = file_header[3].split()[1]
        self.header["bytes_per_sample"] = int(re.findall('(\d*)\s*[bB]', file_header[6])[0])
        self.header["samples"] = int(int(file_header[7].split()[1]))
        self.header["sampling_frequency"] = float(file_header[5].split()[1])
        self._original_samples = int(file_header[7].split()[1])
        duration_micro = 10 ** 6 * float(file_header[7].split()[1]) / float(file_header[5].split()[1])
        self.header["duration"] = datetime.timedelta(microseconds=duration_micro)
        date = file_header[10][18:].strip()

        # we shift the starting time of 18s to go from GPS time to UTC time
        locale.setlocale(locale.LC_TIME, "C")  # ensure we use english months names
        if "." in date:
            self.header["start_date"] = datetime.datetime.strptime(date, "%b %d %H:%M:%S.%f %Y") - datetime.timedelta(
                seconds=18)
        else:
            # handle the case where no milisecond is present
            self.header["start_date"] = datetime.datetime.strptime(date, "%b %d %H:%M:%S %Y") - datetime.timedelta(
                seconds=18)
        self.header["end_date"] = self.header["start_date"] + self.header["duration"]

    def read_data(self):
        sampsize = self.header["bytes_per_sample"]
        with open(self.path, 'rb') as file:
            file.seek(0)  # reset cursor pos
            data = np.fromfile(file, dtype=np.uint8, offset=400)

        data = data[:-(len(data) % sampsize)] if (len(data) % sampsize) != 0 else data
        data = data.reshape((-1, sampsize))  # original array with custom nb of bytes

        next_pow_2 = 2 ** math.ceil(math.log2(sampsize))

        # prepare array of next_pow_2 bytes
        valid_data = np.zeros((data.shape[0], next_pow_2), dtype=np.uint8)
        neg = (data[:, 0] >= 2 ** 7)
        # in case the nb if negative, add several 1 before
        valid_data[:, :next_pow_2 - sampsize] = \
            np.swapaxes(np.full((next_pow_2 - sampsize, neg.shape[0]), (2 ** 8 - 1) * neg), 0, 1)
        valid_data[:, 1:] = data  # copy data content
        # concatenate these bytes to form an int32
        data = np.frombuffer(valid_data[:, ::-1].tobytes(), dtype=np.int32)

        # now convert data to meaningful data
        self.data = scipy.signal.detrend(data) * self.TO_VOLT / 10 ** (self.SENSIBILITY / 20)