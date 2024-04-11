import os
import numpy as np
import pywt
import skimage
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import decimate


class FeaturesExtractor:
    """ General class to transform raw audia signals into various features.
    """
    EXTENSION = ""  # extension of the output file when saving the features

    def __init__(self, manager):
        """ Constructor simply taking a SoundFilesManager.
        :param manager: The SoundFilesManager instance.
        """
        self.manager = manager

    def get_features(self, start, end):
        """ Given a segment, return the features of its content.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: The features of the segment, None in case a part of it is not available.
        """
        data = self.manager.getSegment(start, end)
        return self._get_features(data) if data is not None else None

    def save_features(self, start, end, path):
        """ Given a segment, save the features of its content.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :param path: The path of the file in which we want to save the features.
        :return: None.
        """
        if not os.path.isfile(path):
            self._save_features(self.get_features(start, end), path)

    def save_features_batch(self, starts, ends, paths):
        """ Given a list of segments, save the features of their contents.
        :param start: Start datetimes of the segments.
        :param end: End datetimes of the segments.
        :param paths: The paths of the files in which we want to save the features.
        :return: None.
        """
        for i in range(len(starts)):
            self.save_features(starts[i], ends[i], paths[i])

    def show_features(self, start, end):
        """ Given a segment, show the features of its content.
        :param start: Start datetime of the segment.
        :param end: End datetime of the segment.
        :return: None.
        """
        features = self.get_features(start, end)
        self._show_features(features)

    def _get_features(self, data):
        """ Given raw data, compute the features.
        :param data: Raw data, as an acoustic waveform (i.e. time series of pressures).
        :return: The corresponding features.
        """
        return None

    def _save_features(self, features, path):
        """ Given features, save them.
        :param features: Features we want to save.
        :param path: The path of the file in which we want to save the features.
        :return: None.
        """
        return None

    def _show_features(self, features):
        """ Given features, show them.
        :param features: Features we want to show.
        :return: None.
        """
        return None

class STFTFeaturesExtractor(FeaturesExtractor):
    """ Spectrogram generation features extractor
    """
    EXTENSION = "png"  # spectrograms are saved as grayscale images

    def __init__(self, manager, nperseg=256, overlap=0.5, window="hamming", apply_log=True, f_min=0, f_max=None,
                 vmin=None, vmax=None, cmap="inferno", axis_labels=True):
        """ Constructor of the extractor, initializing various parameters.
        :param manager: The SoundFilesManager instance.
        :param nperseg: The number of points in each segment in the STFT.
        :param overlap: The overlap fraction between different segments.
        :param window: The window function used in the STFT.
        :param apply_log: If True, log-spectrograms are computed.
        :param f_min: Minimum accepted frequency.
        :param f_max: Maximum accepted frequency.
        :param vmin: Minimum accepted value in the spectrogram (smaller values are set to this value).
        :param vmax: Maximum accepted value in the spectrogram (larger values are set to this value).
        :param cmap: The colormap used when showing the spectrograms.
        :param axis_labels: If True, axis labels are displayed when showing the spectrograms.
        """
        super().__init__(manager)
        self.apply_log = apply_log
        self.nperseg = nperseg
        self.overlap = overlap
        self.f_min = f_min
        self.f_max = f_max
        self.vmin = vmin
        self.vmax = vmax
        self.window = window
        self.cmap = cmap
        self.axis_labels = axis_labels

    def _get_features(self, data, normalize=False):
        """ Get the features given raw datapoints. Perform a STFT according to the selected parameters.
        :param data: Data a a time series of pressure.
        :return: The spectrogram along with its frequency and time bins values.
        """
        # compute spectrogram
        f, t, spectro = signal.spectrogram(data, fs=self.manager.sampling_f, nperseg=self.nperseg,
                                           noverlap=int(self.nperseg * self.overlap), window=self.window)
        if self.apply_log:  # compute log-spectrogram
            spectro = 10 * np.log10(spectro + 1e-20)

        # get the frequency bin the closest to self.f_min and remove all that is beneath.
        f_min_idx = (np.abs(self.f_min - f)).argmin()
        spectro = spectro[f_min_idx:]
        f = f[f_min_idx:]

        # same for self.f_max
        if self.f_max:
            f_max_idx = max((np.abs(self.f_max - f)).argmin(), f_min_idx)
            spectro = spectro[:f_max_idx]
            f = f[:f_max_idx]

        # truncate values smaller or larger than respectively self.vmin and self.vmax
        vmin = self.vmin if self.vmin is not None else spectro.min()
        vmax = self.vmax if self.vmax is not None else spectro.max()
        spectro[spectro > vmax] = vmax
        spectro[spectro < vmin] = vmin

        # put the values between 0 and 255 using self.vmin and self.vmax
        spectro = spectro - (self.vmin or 0)
        spectro = 255 * spectro / ((self.vmax or np.max(spectro)) - (self.vmin or 0))

        return f[::-1], t, spectro[::-1]


    def _save_features(self, features, path):
        """ Save the features as a grayscale spectrogram.
        :param features: Spectrogram to save.
        :param path: Path where we want to save the spectrogram.
        :return: None.
        """
        (f, t, spectro) = features

        # choose grayscale with np.uint8 format (that is one byte per pixel)
        skimage.io.imsave(path, spectro.astype(np.uint8))

    def _get_extent_and_aspect(self, features):
        """ Given a spectrogram, get the extent of the axis and the aspect of the image to show a nice rectangle.
        :param features: Spectrogram.
        :return: A couple (extent, aspect).
        """
        (f, t, spectro) = features
        extent = (min(t), max(t), min(f), max(f))
        aspect = (max(t) - min(t)) / (max(f) - min(f) + 100)  # enables to always have a fair rectangle
        return (extent, aspect)

    def _show_features(self, features):
        """ Show a spectrogram acording to the selected parameters.
        :param features: Spectrogram we want to show.
        :return: None.
        """
        (f, t, spectro) = features
        extent, aspect = self._get_extent_and_aspect(features)
        plt.imshow(spectro, extent=extent, aspect=aspect, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        if self.axis_labels:
            plt.xlabel("time (s)")
            plt.ylabel("frequency (Hz)")

class DWTFeaturesExtractor(FeaturesExtractor):
    """ Scalogram generation features extractor.
    """
    EXTENSION = "npy"  # save scalograms as numpy array (not a picture because each level has a different resolution)

    def __init__(self, manager, wavelet='bior2.4', n_levels=8, apply_log=False, vmin=None, vmax=None):
        """ Constructor of the extractor, initializing various parameters.
        :param manager: The SoundFilesManager instance.
        :param wavelet: The type of wavelet we want to use.
        :param n_levels: The number of levels to compute.
        :param apply_log: If True, compute log-scalograms.
        :param vmin: Minimum value to keep in the scalogram.
        :param vmax: Maximum value to keep in the scalogram.
        """
        super().__init__(manager)
        self.n_levels = n_levels
        self.wavelet = wavelet
        self.apply_log = apply_log
        self.vmin = vmin
        self.vmax = vmax

    def _get_features(self, data):
        """ Compute DWT (Discrete Wavelets Transform) given some data points.
        :param data: Data as pressure time series.
        :return: The result of the DWT as a list of lists.
        """
        features = pywt.wavedec(data, self.wavelet, level=self.n_levels)
        for level in range(self.n_levels+1):
            if self.apply_log:
                features[level] = 10 * np.log10(np.abs(features[level]) + 1e-3)
            # truncate values smaller or larger than respectively self.vmin and self.vmax
            vmin = self.vmin if self.vmin is not None else features[level].min()
            vmax = self.vmax if self.vmax is not None else features[level].max()
            features[level][features[level] > vmax] = vmax
            features[level][features[level] < vmin] = vmax
        return features

    def _save_features(self, features, path):
        """ Save features as a simple npy file.
        :param features: DWT result.
        :param path: Path where we want to save the features.
        :return: None.
        """
        np.save(path, features)

    def _show_features(self, features):
        """ Show a scalogram, duplicating values in each scale such that each scale has the same number of values.
        :param features: DWT result.
        :return: None.
        """
        data = np.zeros((len(features), len(features[-1])))
        data[0] = abs(np.repeat(features[0], 2 ** (len(features) - 2))[:len(features[-1])])

        # repeat each point in each scale to match the largest size
        for level in range(1, self.n_levels+1):
            data[level] = abs(np.repeat(features[level], 2 ** (len(features) - level - 1))[:len(features[-1])])

        aspect = (len(data[0])) / (self.n_levels + 10)  # enables to always have a fair rectangle
        plt.imshow(data[::-1], aspect=aspect, vmin=self.vmin, vmax=self.vmax)
        plt.colorbar()

class RelativeDWTFeaturesExtractor(FeaturesExtractor):
    """ Relative DWT features extractor. The idea is to save a segment with the proporition of energy of each of its
    scale after a DWT transform.
    """
    EXTENSION = "npy"  # numpy format

    def __init__(self, manager, wavelet='bior2.4', n_levels=8):
        """ Constructor of the extractor, initializing various parameters.
        :param manager: The SoundFilesManager instance.
        :param wavelet: The type of wavelet we want to use.
        :param n_levels: The number of levels to compute.
        """
        super().__init__(manager)
        self.n_levels = n_levels
        self.wavelet = wavelet

    def save_features_batch_single_file(self, starts, ends, path):
        """ Save the features of several segments at once, in a single file.
        :param starts: Starts of the segments.
        :param ends: Ends of the segments.
        :param path: Path where we want to save the features.
        :return: None.
        """
        if not os.path.isfile(path):
            res = []
            for start, end in zip(starts, ends):
                res.append(self.get_features(start, end))
            np.save(path, np.array(res))

    def _get_features(self, data):
        """ Compute relative DWT levels values given some data points.
        :param data: Data as pressure time series.
        :return: The features as a numpy array.
        """
        features = pywt.wavedec(data, self.wavelet, level=self.n_levels)
        s = [_ for _ in range(self.n_levels)]
        for k in range(0, self.n_levels):
            s[k] = np.sum(np.abs(features[k])) / len(features[k])  # average value of each level
        S = np.array(s)/(np.sum(s))  # contribution of each level in the sum of the everage values of each level
        return S

    def _save_features(self, features, path):
        """ Save some features as a npy file.
        :param features: The features to save.
        :param path: The path where we want to save our features.
        :return: None.
        """
        np.save(path, features)

    def _show_features(self, features):
        """ Show the features as curve.
        :param features: The features we want to show as a numpy array.
        :return: None.
        """
        plt.plot(features)

class WaveformDataFeaturesExtractor(FeaturesExtractor):
    """ Waveform features extractor. Simply take the raw data and enable to plot/save it.
    """
    EXTENSION = "npy"  # numpy format

    def __init__(self, manager, downsampling_factor=1):
        """ Constructor of the extractor, initializing various parameters.
        :param manager: The SoundFilesManager instance.
        :param downsampling_factor: Down-sampling factor in case we want to decimate the signal.
        """
        super().__init__(manager)
        self.downsampling_factor = downsampling_factor

    def _get_features(self, data):
        """ Simply decimate the raw waveform and return it. Consider int32 types.
        :param data: Raw weveform as pressure time series.
        :return: The decimated signal as a numpy array.
        """
        data = decimate(data, self.downsampling_factor).astype(np.int32)
        return data

    def _save_features(self, features, path):
        """ Save the features in a npy file.
        :param features: Decimated signal as a numpy array.
        :param path: Path where we want to save the features.
        :return: None.
        """
        np.save(path, features)

    def _show_features(self, features):
        """ Plot the signal.
        :param features: Decimated signal as a numpy array.
        :return: None.
        """
        plt.plot(features)
