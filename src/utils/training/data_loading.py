import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from tqdm import tqdm


def lines_to_line_generator(csv_lines, repeat=True):
    """ Take a list of lines from a csv summing up a dataset and generate the description of the samples line per line.
    :param csv_lines: List of lines, where each line is a list of the form [sample_path, class, offset_1, offset_2,...]
    where offset_i is the position of the signal number i in seconds relatively to the start time of the sample.
    :param repeat: If True, the generator will loop through the lines and never reach an end.
    :return: A list of the form [sample_path, offset_1, offset_2,...] for each line of csv_lines.
    """
    while True:
        for line in csv_lines:
            res = [line[0]]
            res.extend(line[2:])
            yield tuple(res)
        if not repeat:
            return


def load_spectro(file_path, size, channels):
    """ Read a spectrogram from a .png file and prepare it to run with Tensorflow.
    :param file_path: The path of the image to load.
    :param size: The wanted size of the image with the form (height, width).
    :param channels: The number of channels of the image (1 for grayscale, 3 for RGB).
    :return: The loaded image, resized according to size, as a Tensorflow tensor.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=channels)  # RGB or grayscale
    img = tf.image.convert_image_dtype(img, tf.uint8)  # 0-255 pixels
    img = tf.image.resize(img, size=size)  # resize to correct input size
    return img


def get_line_to_spectro_seg(size, duration_s, channels=1, objective_curve_width=10):
    """ Factory creating a function line_to_specto_seg that loads spectrograms and create the associated ground truth
    curves for Tensorflow.
    :param size: The size of the spectrogram with the form (height, width).
    :param duration_s: The duration, in seconds, of each spectrogram.
    :param channels: The number of channels of the image (1 for grayscale, 3 for RGB).
    :param objective_curve_width: The half width, in seconds, of the abs function in the objective curve at each event.
    :return: A function that loads spectrograms and create the associated ground truth curves.
    """
    time_res = duration_s / size[1]
    objective_curve_width = objective_curve_width / time_res

    def line_to_spectro_seg(data):
        """ Function that takes some input_data as lists and outputs couples (images, ground_truth_curves) for
        Tensorflow.
        :param data: List of form [path, offset_1, offset_2,...] where path is the path of the spectrogram file
        and offset_i is the position of the signal number i in seconds relatively to the start time of the sample.
        :return: A couple (images, ground_truth_curves) made of tensors.
        """
        file_path = data[0]
        events = data[1:]
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
            # use this distance to make "triangles" around events
            y_list.append(tf.math.maximum((objective_curve_width - d) / objective_curve_width, tf.zeros(1)))

        y = tf.stack(y_list)  # obtain the tensor containing all assigned ground truth values
        y = tf.reshape(y, (size[1],))
        return img, y

    return line_to_spectro_seg


def get_line_to_dataset_waveform(size, duration_s, objective_curve_width=10, min_val=10 ** (-35 / 20),
                                 max_val=10 ** (140 / 20)):
    """ Factory creating a function line_to_dataset_waveform that loads waveforms and create the associated ground truth
    curves.
    :param size: The size of the waveform as a number of points.
    :param duration_s: The duration, in seconds, of each sample.
    :param objective_curve_width: The half width, in seconds, of the abs function in the objective curve at each event.
    :param min_val: The minimal accepted value, putting smaller values to 0.
    :param max_val: The maximal accepted value, putting larger values to 1.
    :return: A function that loads waveforms and create the associated ground truth curves.
    """
    time_res = duration_s / size
    objective_curve_width = objective_curve_width / time_res

    def line_to_dataset_waveform(data):
        """ Function that takes some input_data as lists and outputs couples (waveforms, ground_truth_curves).
        :param input_data: List of form [path, offset_1, offset_2,...] where path is the path of the waveform file
        and offset_i is the position of the signal number i in seconds relatively to the start time of the sample.
        :return: A couple (waveforms, ground_truth_curves).
        """
        X, Y = [], []
        for d in tqdm(data):
            waveform = np.load(d[0])[:size]

            target_indices = np.linspace(0, len(waveform) - 1, size)
            interp_f = interp1d(np.arange(len(waveform)), waveform, kind='linear', fill_value="extrapolate")
            waveform = interp_f(target_indices)

            # put the input object in the range [0, 1] according to the accepted range
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
    """ Factory creating a function load_spectro_for_class that loads spectrograms with their expected class.
    :param size: The size of the spectrogram as a side length.
    :param channels: The number of channels of the image (1 for grayscale, 3 for RGB).
    :return:
    """

    def load_spectro_for_class(file_path):
        """ Function that takes some file_path and outputs couples (waveforms, ground_truth_curves).
        :param file_path: The path of the spectrogram file.
        :return: A couple (waveforms, class) made of tensors with class belonging to {0, 1}.
        """
        img = load_spectro(file_path, (size, size), channels)

        if tf.strings.regex_full_match(file_path, ".*negatives.*"):
            return img, 0
        else:
            return img, 1

    return load_spectro_for_class
