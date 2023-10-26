import csv

import tensorflow as tf
import tensorflow_probability as tfp


def csv_to_lines(file, col_to_keep=-1):
    with open(file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        lines = list(csv_reader)
    pos = [l for l in lines if l[1] == "positive"]
    neg = [l for l in lines if l[1] == "negative"]
    if col_to_keep >= 0:
        pos = [p[col_to_keep] for p in pos]
        neg = [n[col_to_keep] for n in neg]
    return pos, neg


def lines_to_line_generator(csv_lines):
    while True:
        for line in csv_lines:
            res = [line[0]]
            res.extend(line[2:])
            yield tuple(res)

def load_spectro(file_path, size, channels):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=channels)  # RGB or grayscale
    img = tf.image.convert_image_dtype(img, tf.uint8)  # 0-255 pixels
    img = tf.image.resize(img, size=size)  # resize to correct input size
    return img


def get_line_to_spectro_seg(size, duration_s, channels=1, gaussian_stdvar_s=10):
    time_res = duration_s / size[1]
    gaussian_stdvar = gaussian_stdvar_s / time_res

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
            events_array = events_array.write(e, (duration_s / 2 + tf.strings.to_number(event)) / time_res)
        events = events_array.stack()

        distrib = tfp.distributions.Normal(loc=0, scale=gaussian_stdvar)
        y_list = []  # list of tensors that each represent 1 time step, will be concatenated as function output
        for i in range(size[1]):
            d = tf.math.reduce_min(
                tf.abs(float(i) - events))  # "distance" in pixels separating this time step from the closest event
            y_list.append(
                distrib.prob(d) / distrib.prob(0))  # assigned ground truth of this time step, normalized to [0,1]

        y = tf.stack(y_list)  # obtain the tensor containing all assigned ground truth values
        return img, y

    return line_to_spectro_seg


def get_line_to_spectro_class(size, channels=1):
    def line_to_spectro_class(file_path):
        img = load_spectro(file_path, size, channels)

        if tf.strings.regex_full_match(file_path, ".*negative.*"):
            return img, 0
        else:
            return img, 1

    return line_to_spectro_class