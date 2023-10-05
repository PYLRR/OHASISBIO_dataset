import datetime
import wave

import scipy
import numpy as np
import math
import re
import locale

class SoundFile:
    EXTENSION = ""
    MAX_VALUE = 2**23-1

    # path: path of the file which we will look for
    # skip_data: if True, only the header of the file will be read, so that it is a quick operation
    def __init__(self, path, skip_data=False):
        self.header = {}
        self.data = []
        self.path = path
        self.read_header()
        if not skip_data:
            self.read_data()

    def read_header(self):
        pass

    # to use when the file was previously header-ony loaded but we now want its data
    def read_data(self):
        pass

    # load the data of the file but enable to only read after and before a datetime
    def get_data(self, start=None, end=None):
        data = self.data
        if start is not None and end is not None:
            assert end > start, "required end is before or equal to the start"
        if start is not None:
            offset = start - self.header["start_date"]
            offset_points = int(offset.total_seconds() * self.header["sampling_frequency"])
            if offset_points > 0:
                data = data[offset_points:]
        if end is not None:
            offset = self.header["end_date"] - end
            offset_points = int(offset.total_seconds() * self.header["sampling_frequency"])
            if offset_points > 0:
                data = data[:-offset_points]
        return data

    def write_wav(self, path):
        to_write = np.int32((2**15-1) * self.data / self.MAX_VALUE)
        scipy.io.wavfile.write(path, int(self.header["sampling_frequency"]), to_write)

    # overrides == operator with str so that we can look for a file in a list with its path
    def __eq__(self, other):
        if type(other) == str:
            return self.path == other
        if type(other) == type(self):
            return self.path == other.path
        return False

class WavFile(SoundFile):
    EXTENSION = "wav"

    def read_header(self):
        with wave.open(self.path, 'rb') as file:
            self.header["sampling_frequency"] = file.getframerate()
            self.header["samples"] = file.getnframes()

        duration_micro = 10 ** 6 * self.header["samples"] / self.header["sampling_frequency"]
        self.header["duration"] = datetime.timedelta(microseconds=duration_micro)
        file_name = self.path.split("/")[-1][:-4]  # get the name of the file and get rid of extension
        self.header["start_date"] = datetime.datetime.strptime(file_name, "%Y%m%d_%H%M%S")
        self.header["end_date"] = self.header["start_date"] + self.header["duration"]

    def read_data(self):
        _, self.data = scipy.io.wavfile.read(self.path)