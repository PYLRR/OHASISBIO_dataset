import glob
import math
import os
from collections import deque

import numpy as np
from tqdm import tqdm

from utils.data_reading.sound_file import SoundFile, WavFile

# this class provides an abstraction of the file reading part. Given a path containing data files, it enables to ask
# for any time segment without having to deal with problems such as files borders
class SoundFileManager:
    FILE_CLASS = SoundFile

    # path: path of directory containing the files which we will look for
    # cache_size: number of files to keep loaded in memory, so that if one is used again it is fast to load (FIFO)
    def __init__(self, path, cache_size=4):
        self.path = path
        self._initialize_files()

        # cache that keeps most recent files in mem, s.t. they can be used again quicker
        self.cache_size = cache_size
        self.cache = deque()

        # we consider the name of the directory as a station name
        self.path = self.path.rstrip("/")
        self.station_name = self.path.split("/")[-1]

        self._initialize_from_header()

    def _initialize_files(self):
        # open the first file to extract starting time
        files = glob.glob(self.path + "/*." + self.FILE_CLASS.EXTENSION)
        files.sort()
        if len(files) == 0:
            raise Exception(f"No files found in {self.path}")
        self.files = files

    def _initialize_from_header(self):
        self.first_file_number, self.last_file_number = self._findFirstAndLastFileNumber()
        # open the first file to extract some of its properties
        first_file = self.FILE_CLASS(self.files[self.first_file_number], skip_data=True)

        # now look for the last file
        last_file = self.FILE_CLASS(self.files[self.last_file_number], skip_data=True)

        self.dataset_start = first_file.header["start_date"]
        self.sampling_f = first_file.header["sampling_frequency"]
        # in case the dataset begins or finishes with a smaller file, we take the biggest one as reference
        self.files_duration = max(first_file.header["duration"], last_file.header["duration"])
        self.dataset_end = last_file.header["end_date"]

    def _findFirstAndLastFileNumber(self):
        i = 0
        while os.path.getsize(self.files[i]) == 0:
            i += 1

        j = 1
        while os.path.getsize(self.files[-j]) == 0:
            j += 1
        return i, len(self.files) - j

    # get the path of the data file having the number file_number
    def getPath(self, file_number):
        return self.files[file_number]

    # read, add to cache and return a file given its file number, reading only header if skip_data is True
    def loadFile(self, file_number, skip_data=False):
        if file_number in self.cache:
            if len(self.cache[self.cache.index(file_number)].data) == 0 and not skip_data:
                self.cache[self.cache.index(file_number)].read_data()
            return self.cache[self.cache.index(file_number)]
        if len(self.cache) == self.cache_size:
            self.cache.popleft()
        file = self.FILE_CLASS(self.getPath(file_number), skip_data=skip_data)
        self.cache.append(file)
        return file

    # loads a file containing a given datetime, computed by comparison with the first file of the directory or on a
    # given file to ensure exactness.
    # Indeed, clock drift or float imprecision can lead for example to load a file starting slightly after the event,
    # so if this happens we use this file starting date metadata to compute the right file (likely the one just before)
    # To provide a reference file, it must be set in parameter ref in format (ref_date,ref_file_number)
    def locateAndLoadFile(self, target_datetime, ref=(None, None)):
        if ref == (None, None):
            ref = (self.dataset_start, self.first_file_number)

        difference = target_datetime - ref[0]
        file_nb = ref[1] + difference / self.files_duration
        file_nb = min(file_nb, self.last_file_number)  # ensure we don't get out of the dataset because of clock drift

        file = self.loadFile(int(file_nb), skip_data=True)
        offset = target_datetime - file.header["start_date"]

        # if target date is not indeed in this file, use this file start as a ref to look for the good one
        if offset.total_seconds() < 0 or file.header["end_date"] < target_datetime:
            return self.locateAndLoadFile(target_datetime, ref=(file.header["start_date"], int(file_nb)))
        file_nb = int(file_nb) + offset.total_seconds() / file.header["duration"].total_seconds()
        return file_nb

        # given a start date and an end date, return an array containing all the data points between them

    def getSegment(self, start, end):
        first_file, last_file = self._getFilesToLoadFromSegment(start, end)

        file_numbers = range(first_file, last_file + 1)
        data = np.empty(1, dtype=np.int64)
        for file_number in file_numbers:
            file = self.loadFile(file_number)
            file_data = file.get_data(start=start, end=end)
            data = np.concatenate((data, file_data), dtype=file_data.dtype)

        if len(data) == 0:
            print(f"0-length data fetched for files {file_numbers} from date {start} to {end}")

        return data

    def getGenerator(self, start, end, batch_size=1):
        first_file, last_file = self._getFilesToLoadFromSegment(start, end)

        file_numbers = range(first_file, last_file+1)
        points = []
        for file_number in file_numbers:
            file = self.loadFile(file_number)
            data = file.get_data(start=start, end=end)
            i = 0 # 2639980 in data
            while i<len(data):
                up_bound = min(len(data), i+batch_size-len(points))
                points.extend(data[i:up_bound])
                # yield points iff we have a batch, otherwise it means we reached end of a file and we wait the next one
                if len(points) == batch_size:
                    yield points
                    points = []
                i = up_bound
        if len(points) > 0:
            yield points

    def _getFilesToLoadFromSegment(self, start, end):
        assert start >= self.dataset_start, "start is before the first file"
        assert end <= self.dataset_end, "end is after the last file"

        first_file = self.locateAndLoadFile(start)
        segment_length = end - start
        nb_files = segment_length / self.files_duration
        # get last file index, considering the last file must have at least 1 data sample
        last_file = math.floor(first_file + nb_files - 1 / self.files_duration.total_seconds())
        first_file = math.floor(first_file)
        return first_file, last_file

    def flushCache(self):
        self.cache.clear()

    def writeWav(self, dir_path, index_file_name="index", write_index_only=False):
        index_file = dir_path + index_file_name + ".csv"

        if not os.path.isfile(index_file) or write_index_only:
            with open(index_file, "w+") as f:
                f.write("station, segment_start, segment_end, sampling_f, path\n")

        for fn in tqdm(range(self.last_file_number+1)):
            name = self.getPath(fn).split("/")[-1][:-4]  # get the file name without extension (e.g. "00001")
            out_path = dir_path + name + ".wav"

            if write_index_only:
                file = self.loadFile(fn, skip_data=True)
            else:
                if os.path.isfile(out_path):
                    continue
                file = self.loadFile(fn)
                file.write_wav(out_path)

            with open(index_file, "a") as f:
                f.write('%(station)s, %(segment_start)s, %(segment_end)s, %(sampling_f)s, %(path)s\n' %
                        {"station": file.header["site"],
                         "segment_start": file.header["start_date"].strftime("%Y%m%d-%H%M%S"),
                         "segment_end": file.header["end_date"].strftime("%Y%m%d-%H%M%S"),
                         "sampling_f": file.header["sampling_frequency"],
                         "path": out_path})

    def __eq__(self, other):
        if type(self) == type(other) and other.path == self.path:
            return True
        return False
class WavFilesManager(SoundFileManager):
    FILE_CLASS = WavFile
