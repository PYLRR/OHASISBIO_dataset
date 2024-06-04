import numpy as np
from scipy.signal import butter, sosfilt
def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Create a Butterworth bandpass filter.
    :param lowcut: The low cutting frequency in Hz.
    :param highcut: The high cutting frequency in Hz.
    :param fs: The sampling frequency in Hz.
    :param order: The order of the Butterworth filter.
    :return: The Butterworth filter.
    """
    return butter(order, [lowcut, highcut], fs=fs, btype='band', analog=False, output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, pad_s=10):
    """ Create a Butterworth bandpass filter and apply it on the provided data.
    :param data: Input data to filter.
    :param lowcut: The low cutting frequency in Hz.
    :param highcut: The high cutting frequency in Hz.
    :param fs: The sampling frequency in Hz.
    :param order: The order of the Butterworth filter.
    :param pad_s: The duration, in s, to add before the data to avoid filter border effects. The corresponding number
    of points is added, taking the mean of the data as a value. These points are removed from output.
    :return: The filtered data.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    data = [np.mean(data[:pad_s*int(fs)])] * pad_s * int(fs) + list(data)
    y = sosfilt(sos, data)[pad_s * int(fs):]
    return y
