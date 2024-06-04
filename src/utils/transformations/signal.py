import numpy as np
from scipy.signal import butter, lfilter, sosfilt
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', analog=False, output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, pad_s=10):
    # pad_s is the number of seconds added at the start of the time series to avoid border effect
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    data = [np.mean(data[:pad_s*int(fs)])] * pad_s * int(fs) + list(data)
    y = sosfilt(sos, data)[pad_s * int(fs):]
    return y
