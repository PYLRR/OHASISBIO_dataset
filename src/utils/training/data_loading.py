import csv

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.interpolate import interp1d
from tqdm import tqdm


def lines_to_line_generator(csv_lines, repeat=True):
    while True:
        for line in csv_lines:
            res = [line[0]]
            res.extend(line[2:])
            yield tuple(res)
        if not repeat:
            return


def load_spectro(file_path, size, channels):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=channels)  # RGB or grayscale
    img = tf.image.convert_image_dtype(img, tf.uint8)  # 0-255 pixels
    img = tf.image.resize(img, size=size)  # resize to correct input size
    return img


def get_line_to_spectro_seg(size, duration_s, channels=1, objective_curve_width=10):
    time_res = duration_s / size[1]
    objective_curve_width = objective_curve_width / time_res

    def line_to_spectro_seg(input_data):
        file_path = input_data[0]
        events = input_data[1:]
        nb_events = int(len(events))

        img = load_spectro(file_path, size, channels)

        if nb_events == 0:
            # if the sample is negative we consider it contains no signal of interest
            return img, tf.zeros(size[1])

        # we have a positive sample, we extract the location of the signals of interest in it
        events_array = tf.TensorArray(tf.float32, size=nb_events, clear_after_read=False)
        for e in range(nb_events):
            event = events[e]
            # now get the idx of the x-axis of the image where the event occurs
            pos = (duration_s / 2 + tf.strings.to_number(event)) / time_res
            events_array = events_array.write(e, pos)
        events = events_array.stack()

        y_list = []  # list of tensors that each represent 1 time step, will be concatenated as function output
        for i in range(size[1]):
            d = tf.math.reduce_min(
                tf.abs(float(i) - events))  # "distance" in pixels separating this time step from the closest event
            y_list.append(tf.math.maximum((objective_curve_width - d) / objective_curve_width, tf.zeros(1)))

        y = tf.stack(y_list)  # obtain the tensor containing all assigned ground truth values
        y = tf.reshape(y, (size[1],))
        return img, y

    return line_to_spectro_seg


def get_line_to_dataset_waveform(size, duration_s, objective_curve_width=10, min_val=10 ** (-35 / 20),
                                 max_val=10 ** (140 / 20)):
    time_res = duration_s / size
    objective_curve_width = objective_curve_width / time_res

    def line_to_dataset_waveform(data):
        X, Y = [], []
        for d in tqdm(data):
            waveform = np.load(d[0])[:size]

            target_indices = np.linspace(0, len(waveform) - 1, size)
            interp_f = interp1d(np.arange(len(waveform)), waveform, kind='linear', fill_value="extrapolate")
            waveform = interp_f(target_indices)

            waveform[np.abs(waveform) < min_val] = min_val
            waveform[np.abs(waveform) > max_val] = max_val
            waveform[waveform > 0] = (waveform[waveform > 0] - min_val) / max_val
            waveform[waveform < 0] = (waveform[waveform < 0] + min_val) / max_val

            X.append(np.array(waveform, dtype=np.float32))
            events = d[2:]
            nb_events = int(len(events))

            if nb_events == 0:
                # if the sample is negative we consider it contains no signal of interest
                Y.append(np.zeros(size, dtype=np.float16))
                continue

            # we have a positive sample, we extract the location of the signals of interest in it
            events = [(duration_s / 2 + float(e)) / time_res for e in events]

            res = np.arange(size)
            # get the distance between each time step and the closest event from events array
            res = np.min(np.abs(np.subtract.outer(res, events)), axis=1)
            # use this distance to make "triangles" around events
            res = ((objective_curve_width - res) / objective_curve_width).astype(np.float16)
            res[res < 0] = 0

            Y.append(res)
        return np.array(X), np.array(Y)

    return line_to_dataset_waveform


def get_load_spectro_for_class(size=128, channels=1):
    def load_spectro_for_class(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=channels)  # RGB
        img = tf.image.convert_image_dtype(img, tf.uint8)  # 0-255 pixels
        img = tf.image.resize(img, size=(size, size))  # resize to standard resnet input size

        if tf.strings.regex_full_match(file_path, ".*negative.*"):
            return img, 0
        else:
            return img, 1
    return load_spectro_for_class
