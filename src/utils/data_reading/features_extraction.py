import datetime
import os

import numpy as np
import pywt
import skimage
import yaml
from matplotlib import pyplot as plt
from scipy import signal
from tqdm import tqdm

from utils.data_reading.sound_file_manager import WavFilesManager


class FeaturesExtractor:
    EXTENSION = ""

    def __init__(self, manager):
        self.manager = manager

    def get_features(self, start, end):
        return self._get_features(self.manager.getSegment(start, end))

    def save_features(self, start, end, path):
        if not os.path.isfile(path):
            self._save_features(self.get_features(start, end), path)

    def save_features_batch(self, starts, ends, paths):
        for i in range(len(starts)):
            self.save_features(starts[i], ends[i], paths[i])

    def show_features(self, start, end):
        features = self.get_features(start, end)
        self._show_features(features)

    def _get_features(self, data):
        return None

    def _save_features(self, features, path):
        return None

    def _show_features(self, features):
        return None

class STFTFeaturesExtractor(FeaturesExtractor):
    EXTENSION = "png"

    def __init__(self, manager, nperseg=256, overlap=0.5, apply_log=True, f_min=0, f_max=None, vmin=None, vmax=None,
                 window="hamming"):
        super().__init__(manager)
        self.apply_log = apply_log
        self.nperseg = nperseg
        self.overlap = overlap
        self.f_min = f_min
        self.f_max = f_max
        self.vmin = vmin
        self.vmax = vmax
        self.window = window

    def _get_features(self, data):
        f, t, spectro = signal.spectrogram(data, fs=self.manager.sampling_f, nperseg=self.nperseg,
                                           noverlap=int(self.nperseg * self.overlap), window=self.window)
        if self.apply_log:
            spectro = 10 * np.log10(spectro + 1e-20)


        f_min_idx = (np.abs(self.f_min - f)).argmin()
        spectro = spectro[f_min_idx:]
        f = f[f_min_idx:]
        if self.f_max:
            f_max_idx = max((np.abs(self.f_max - f)).argmin(), f_min_idx)
            spectro = spectro[:f_max_idx]
            f = f[:f_max_idx]

        vmin = self.vmin if self.vmin is not None else spectro.min()
        vmax = self.vmax if self.vmax is not None else spectro.max()
        spectro[spectro > vmax] = vmax
        spectro[spectro < vmin] = vmin

        return f[::-1], t, spectro[::-1]

    def _save_features(self, features, path):
        (f, t, spectro) = features

        # put the values between 0 and 255 before saving them
        spectro = spectro - self.vmin
        spectro = 255 * spectro / (self.vmax - self.vmin)

        skimage.io.imsave(path, spectro.astype(np.uint8))

    def _get_extent_and_aspect(self, features):
        (f, t, spectro) = features
        extent = (min(t), max(t), min(f), max(f))
        aspect = (max(t) - min(t)) / (max(f) - min(f) + 100)  # enables to always have a fair rectangle
        return (extent, aspect)

    def _show_features(self, features):
        (f, t, spectro) = features
        extent, aspect = self._get_extent_and_aspect(features)
        plt.imshow(spectro, extent=extent, aspect=aspect, vmin=self.vmin, vmax=self.vmax)
        plt.xlabel("time (s)")
        plt.ylabel("frequency (Hz)")

class DWTFeaturesExtractor(FeaturesExtractor):
    EXTENSION = "npy"

    def __init__(self, manager, wavelet='bior2.4', n_levels=8, apply_log=False, vmin=None, vmax=None):
        super().__init__(manager)
        self.n_levels = n_levels
        self.wavelet = wavelet
        self.apply_log = apply_log
        self.vmin = vmin
        self.vmax = vmax

    def _get_features(self, data):
        features = pywt.wavedec(data, self.wavelet, level=self.n_levels)
        if self.apply_log:
            for level in range(self.n_levels+1):
                features[level] = 10 * np.log10(np.abs(features[level]) + 1e-3)
                vmin = self.vmin if self.vmin is not None else features[level].min()
                vmax = self.vmax if self.vmax is not None else features[level].max()
                features[level][features[level] > vmax] = vmax
                features[level][features[level] < vmin] = vmax
        return features

    def _save_features(self, features, path):
        np.save(path, features)

    def _show_features(self, features):
        data = np.zeros((len(features), len(features[-1])))
        data[0] = abs(np.repeat(features[0], 2 ** (len(features) - 2))[:len(features[-1])])

        for level in range(1, self.n_levels+1):
            data[level] = abs(np.repeat(features[level], 2 ** (len(features) - level - 1))[:len(features[-1])])

        aspect = (len(data[0])) / (self.n_levels + 10)  # enables to always have a fair rectangle
        plt.imshow(data[::-1], aspect=aspect, vmin=self.vmin, vmax=self.vmax)
        plt.colorbar()

class RelativeDWTFeaturesExtractor(FeaturesExtractor):
    EXTENSION = "npy"

    def __init__(self, manager, wavelet='bior2.4', n_levels=8):
        super().__init__(manager)
        self.n_levels = n_levels
        self.wavelet = wavelet

    def save_features_batch(self, starts, ends, paths):
        res = []
        for start, end in tqdm(zip(starts, ends)):
            res.append(self.get_features(start, end))
        np.save(paths[0], np.array(res))

    def _get_features(self, data):
        features = pywt.wavedec(data, self.wavelet, level=self.n_levels)
        s = [_ for _ in range(self.n_levels)]
        for k in range(0, self.n_levels):
            s[k] = np.sum(np.abs(features[k])) / len(features[k])
        S = s/(np.sum(s))
        return S

    def _save_features(self, features, path):
        np.save(path, features)

    def _show_features(self, features):
        plt.plot(features)

class RawDataFeaturesExtractor(FeaturesExtractor):
    EXTENSION = "npy"

    def __init__(self, manager):
        super().__init__(manager)

    def _get_features(self, data):
        return data

    def _save_features(self, features, path):
        np.save(path, features)

    def _show_features(self, features):
        plt.plot(features)
