import datetime
import glob
import math
import os
from collections import deque

import numpy as np
import scipy
from tqdm import tqdm

from utils.data_reading.sound_data.sound_file import SoundFile, WavFile, DatFile, WFile, NpyFile

# epsilon to compare two close datetimes
TIMEDELTA_EPSILON = datetime.timedelta(microseconds=10**4)

class SoundFilesManager:
    """ Class embodying an abstract layer between the sound files and the user such that she can call methods to get the
    data on some periods without worrying about the local organization of the files (e.g. duration of files, what if the
    required period is split on several files...).
    We expect the audio data to be organized as a directory (which is named as its station) of individual sound files.
    Each sound file can give the information of its starting time, its sampling frequency and its length
    (this can be encoded in its header, or simply its name). Their names must be such that when sorted by them, the
    files have to be chronologically sorted.
    This way, the manager will be able to determine which file is responsible for which datetime.
    """
    FILE_CLASS = SoundFile  # type of the individual sound files

    def __init__(self, path, cache_size=4, fill_empty_with_zeroes=True, kwargs=None):
        """ Constructor of the class that reads some files of the folder to get its global organization.
        :param path: Path of the directory containing the files that we want to manage.
        :param cache_size: Number of files to keep loaded in memory, so that if one is used again by the user
        :param fill_empty_with_zeroes: If True, any missing data point between the dataset bounds is replaced by a 0
        :param kwargs: Other arguments as a dict, that can be used by particular FilesManager (e.g. sensitivity...).
        it is fast to load (FIFO fashion).
        """
        self._process_kwargs(kwargs)
        self.path = path
        # get the files in the folder
        self._initialize_files()

        # decimal standard coordinates of the station in (lat, lon) format
        self.coord = (None, None)

        # cache that keeps most recent files in mem, s.t. they can be used again quicker
        self.cache_size = cache_size
        self.cache = deque()

        self.fill_empty_with_zeroes = fill_empty_with_zeroes

        # we consider the name of the directory as a station name
        self.path = self.path.rstrip("/")
        self.station_name = self.path.split("/")[-1]

        # determine some properties of the dataset (covered period, sampling frequencies, ...)
        self._initialize_from_header()

    def _process_kwargs(self, kwargs):
        """ Do something with the available kwargs. (e.g. initialize sensitivity)
        :param kwargs: Dict of additional arguments.
        :return: None.
        """
        pass

    def _initialize_files(self):
        """ Get the list of the files in the directory.
        :return: None.
        """
        files = glob.glob(self.path + "/*." + self.FILE_CLASS.EXTENSION)
        files.sort()  # we assume alphanumerically sorted files are also chronologically sorted
        if len(files) == 0:
            raise Exception(f"No files found in {self.path}")
        self.files = files

    def _initialize_from_header(self):
        """ Get information on the dataset by reading some files of the directory.
        :return: None.
        """
        # get the first and last files indexes
        self.first_file_number, self.last_file_number = self._findFirstAndLastFileNumber()
        # open the first file to extract some of its properties
        first_file = self._initialize_sound_file(self.files[self.first_file_number], True, self.first_file_number)
        # now look for the last file
        last_file = self._initialize_sound_file(self.files[self.last_file_number], True, self.last_file_number)

        # then read the properties of the dataset from the files
        self.dataset_start = first_file.header["start_date"]
        self.dataset_end = last_file.header["end_date"]
        # we assume a similar sf for the whole dataset
        self.sampling_f = first_file.header["sampling_frequency"]
        # we also assume a similar file duration for the whole dataset
        # in case the dataset begins or finishes with a smaller file, we take the biggest one as reference
        self.files_duration = max(first_file.header["duration"], last_file.header["duration"])

    def _findFirstAndLastFileNumber(self):
        """ Find the indexes of the first and last non-empty files of the dataset.
        :return: The indexes (in the files list) of the first and last non-empty files of the dataset.
        """
        i = 0
        while os.path.getsize(self.files[i]) == 0:
            i += 1

        j = 1
        while os.path.getsize(self.files[-j]) == 0:
            j += 1
        return i, len(self.files) - j

    def _getPath(self, file_number):
        """ Given an index of a file, return its path.
        :param file_number: The index of the file to get.
        :return: The path of the file.
        """
        return self.files[file_number]

    def _initialize_sound_file(self, path, skip_data, file_number):
        """ Read a file and return the corresponding SoundFile instance.
        :param path: Path of the file to load.
        :param skip_data: If True, only read metadata.
        :param file_number: Number of the file to use as ID.
        :return: The SoundFile instance.
        """
        return self.FILE_CLASS(path, skip_data=skip_data, identifier=file_number)

    def _loadFile(self, file_number, skip_data=False):
        """ Read, add to cache and return a file given its file number, reading only the metadata if skip_data is True.
        In case the file already exists in cache, it is simply taken from memory (updating the cache).
        :param file_number: The index of the wanted file.
        :param skip_data: Boolean to skip the reading of the file internal data.
        :return: The file as a SoundFile object.
        """
        if file_number in self.cache:
            # cache hit
            cache_idx = self.cache.index(file_number)
            if len(self.cache[cache_idx].data) == 0 and not skip_data:
                self.cache[cache_idx].read_data()
            # put the item in last cache position in case it's not
            if cache_idx != len(self.cache)-1:
                cache_start = deque(list(self.cache)[cache_idx:])
                cache_start.rotate(-1)
                self.cache = deque(list(self.cache)[:cache_idx] + list(cache_start))
            return self.cache[-1]
        # cache miss, in case the cache is full we remove its oldest item
        file = self._initialize_sound_file(self._getPath(file_number), skip_data, file_number)
        if self.cache_size > 0:
            if len(self.cache) == self.cache_size:
                self.cache.popleft()
            self.cache.append(file)
        return file

    def _locateFile(self, target_datetime, ref=(None, None), history=None, round_down=True):
        """ Find a file containing a given datetime and return its index.
        The function is made recursive and enables to use a specific file as a time reference to find the index we look
        for. This is made to face clock drift, float imprecision or other problems leading to a wrong file number
        estimation using the first file and the supposed duration of each file of the dataset.
        :param target_datetime:  The datetime we look for.
        :param ref: The file we use as a time reference to guess the index we look for, format (date,file_number).
        :param history: History of visited files, if we visit two times the same file it means there is a problem.
        :param round_down: If True, choose the previous correct file in case the required time is not in the dataset.
        Otherwise, choose the next one.
        :return: The index of the file we look for, also keeping the decimal part (e.g. 1.5 if the datetime is half the
        way between file 1 and 2), or last valid index before the provided datetime in case it is unavailable.
        """
        # by default, we use the first file of the dataset to guess the index we look for
        if ref == (None, None):
            ref = (self.dataset_start, self.first_file_number)

        # use the files duration to find the expected index of the file containing the required datetime
        difference = target_datetime - ref[0]
        file_nb = ref[1] + difference / self.files_duration
        # ensure we don't get out of the dataset
        file_nb = min(file_nb, self.last_file_number)
        file_nb = max(file_nb, 0)

        # check the file is the good one
        file = self._loadFile(int(file_nb), skip_data=True)
        offset = target_datetime - file.header["start_date"]
        # if target date is not indeed in this file, use this file start as a ref to look for the good one
        if offset.total_seconds() < 0 or file.header["end_date"] < target_datetime:
            file_nb_int = int(file_nb)
            history = history or []
            if file_nb_int in history:
                # a file has been visited twice, which means the required data point is not available
                # we return the last file before the required time, with an exhaustive search
                file_nb = max(history[-1], file_nb_int)
                while self._loadFile(file_nb, skip_data=True).header["start_date"] > target_datetime:
                    file_nb -= 1
                return file_nb
            history.append(file_nb_int)
            return self._locateFile(target_datetime, ref=(file.header["start_date"], int(file_nb)), history=history)
        # else, we're done
        file_nb = int(file_nb) + offset.total_seconds() / file.header["duration"].total_seconds()
        return file_nb

    def _getFilesToLoadFromSegment(self, start, end):
        """ Given a start and end datetime, find the first and last files indexes.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: First and last files indexes.
        """
        assert start >= self.dataset_start - TIMEDELTA_EPSILON, "start is before the first file"
        assert end <= self.dataset_end + TIMEDELTA_EPSILON, "end is after the last file"

        first_file, last_file = self._locateFile(start), self._locateFile(end, round_down=True)
        first_file = math.floor(first_file)
        last_file = math.floor(last_file)
        return first_file, last_file

    def getSegment(self, start, end):
        """ Given a start date and an end date, return an array containing all the data points between them.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: A numpy array containing the data points in the segment, where unavailable points are replaced by 0s.
        """
        end -= TIMEDELTA_EPSILON  # small epsilon to exclude the last point of the interval
        first_file, last_file = self._getFilesToLoadFromSegment(start, end)
        if first_file is None or last_file is None:
            return None

        file_numbers = range(first_file, last_file + 1)
        data = []
        previous_end = start
        for i, file_number in enumerate(file_numbers):
            # if we have no cache, we don't load the file to memory
            file = self._loadFile(file_number, skip_data=(self.cache_size==0))

            # check the file indeed includes wanted data (rounding errors may lead to load a useless file or we may be
            # in a data leap)

            file_data = []
            if file.header["end_date"] > start and file.header["start_date"] < end:
                file_data = file.get_data(start=previous_end, end=end)
                data.extend(file_data)

            next_start = end if i == len(file_numbers) - 1 else self._loadFile(file_numbers[i+1], skip_data=True).header["start_date"]
            # in case a leap is larger than a file, we may have untouched the first file whose end may be before
            # the value of start. In this case, we have to consider only the values after start and pad with 0
            previous_end = file.header["end_date"] if len(file_data) > 0 else previous_end
            if previous_end < next_start and previous_end < end and self.fill_empty_with_zeroes:
                data.extend([0] * np.rint((min(next_start, end) - previous_end).total_seconds() * self.sampling_f).
                            astype(int))

        if len(data) == 0:
            print(f"0-length data fetched for files {file_numbers} from date {start} to {end} (station {self.station_name})")

        return np.array(data)

    def flushCache(self):
        """ Clear the cache.
        :return: None.
        """
        self.cache.clear()

    def toWav(self, start, end, file_duration, path):
        """ Load the required segment and save it as a wav file.
        :param start: Start of the segment to save. Start of the dataset if None.
        :param end: End of the segment to save. End of the dataset if None.
        :param file_duration: Duration, as a datetime.timedelta, of a single file.
        :param path: Path of the directory where to save the wav file.
        :return: None
        """
        start, end = start or self.dataset_start, end or self.dataset_end
        for i in tqdm(range(math.ceil((end-start)/file_duration)), desc="Converting dataset"):
            seg_start = start + i*file_duration
            seg_end = min(seg_start + file_duration, end)

            data = np.array(self.getSegment(seg_start, seg_end), dtype=np.int32)
            scipy.io.wavfile.write(f'{path}/{seg_start.strftime("%Y%m%d_%H%M%S")}.wav',
                                   int(self.sampling_f), data)


    def __eq__(self, other):
        """ Test if another manager works with the same path.
        :param other: Another manager.
        :return: True if other is a manager with the same directory, else False.
        """
        if type(self) == type(other) and other.path == self.path:
            return True
        return False

class WavFilesManager(SoundFilesManager):
    """ Class accounting for .wav files.
    """
    FILE_CLASS = WavFile

class NpyFilesManager(SoundFilesManager):
    """ Class accounting for .npy files.
    """
    FILE_CLASS = NpyFile

    def getSegment(self, start, end):
        data = super().getSegment(start, end)
        data = np.swapaxes(data, 0, 1)
        return data

class DatFilesManager(SoundFilesManager):
    """ Class accounting for .dat files specific of GEO-OCEAN lab.
    """
    FILE_CLASS = DatFile

    def _process_kwargs(self, kwargs):
        """ Initilize sensitivity.
        :param kwargs: Dict of additional arguments.
        :return: None.
        """
        self.sensitivity = float(kwargs["sensitivity"]) \
            if kwargs is not None and "sensitivity" in kwargs else -163.5  # default

    def _initialize_sound_file(self, path, skip_data, file_number):
        """ Read a file and return the corresponding SoundFile instance.
        :param path: Path of the file to load.
        :param skip_data: If True, only read metadata.
        :param file_number: Number of the file to use as ID.
        :return: The SoundFile instance.
        """
        return self.FILE_CLASS(path, self.sensitivity, skip_data=skip_data, identifier=file_number)

class WFilesManager(SoundFilesManager):
    """ Class accounting for .w files specific of CTBTO.
    Virtually, we represent 1 record (1 line of .wfdisc) by 1 file (1 instance of WFile).
    """
    FILE_CLASS = WFile

    def _initialize_files(self):
        """ Get the list of the virtual files (nb or records) in the directory.
        :return: None.
        """
        files = glob.glob(self.path + "/wfdisc/*.wfdisc")
        files.sort()  # we assume alphanumerically sorted files are also chronologically sorted
        if len(files) == 0:
            raise Exception(f"No files found in {self.path}/wfdisc")
        vfiles = []
        for file in files:
            with open(file, "r") as f:
                lines = f.readlines()
            lines = [line.split() for line in lines]
            paths = ["/".join(file.split("/")[:-2])+"/"+line[16] for line in lines]
            starts = [datetime.datetime.utcfromtimestamp(float(line[2])) for line in lines]
            ends = [datetime.datetime.utcfromtimestamp(float(line[6])) for line in lines]
            indexes_start = [int(line[17])//4 for line in lines]
            indexes_end = indexes_start[1:] + [None]
            sfs = [float(line[8]) for line in lines]
            cnt_to_upa = [float(line[9]) for line in lines]
            name = [str(line[0]) for line in lines]
            vfiles.extend(zip(paths, starts, ends, indexes_start, indexes_end, sfs, cnt_to_upa, name))
        self.start_dates = np.array([f[1] for f in vfiles])
        # sort files by date
        argsort = np.argsort(self.start_dates)
        self.start_dates = self.start_dates[argsort]
        self.files = np.array(vfiles)[argsort]

    def _getPath(self, file_number):
        """ Given an index of a vfile, return its path.
        :param file_number: The index of the vfile to get.
        :return: The path of the vfile, its dates of start and end, indexes of start and end, sampling f, count to upa
        parameter and name
        """
        return self.files[file_number]

    def _findFirstAndLastFileNumber(self):
        """ Find the indexes of the first and last vfiles of the dataset.
        :return: The indexes (in the files list) of the first and last vfiles of the dataset.
        """
        return 0, len(self.files) - 1

    def _locateFile(self, target_datetime, ref=(None, None), history=None, round_down=True):
        closest = np.argmin(np.abs(target_datetime - self.start_dates))
        if round_down:
            closest = closest - 1 if target_datetime < self.files[closest][1] else closest
        else:
            closest = closest - 1 if target_datetime < self.files[closest][1] and target_datetime < \
                                     self.files[closest - 1][2] else closest
        return closest

def make_manager(path, kwargs):
    """ Looks for the extension of the files in the given directory and returns the correct files manager.
    :param path: The path of the directory we want to load.
    :param kwargs: Other arguments as a dict, that can be used by particular FilesManager (e.g. sensitivity...).
    :return: A FilesManager instance able to load the files of the given directory, or None.
    """
    files = [f.split('.')[-1] for f in os.listdir(path)]
    if WavFile.EXTENSION in files:
        return WavFilesManager(path, kwargs=kwargs)
    if DatFile.EXTENSION in files:
        return DatFilesManager(path, kwargs=kwargs)
    if WFile.EXTENSION in files:
        return WFilesManager(path, kwargs=kwargs)
    print(f"No matching manager found for path {path}")
    return None