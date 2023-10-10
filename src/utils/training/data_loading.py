import tensorflow as tf
import tensorflow_probability as tfp

SPECTRO_SIZE_SEG = (128, 128)
SPECTRO_DURATION_S = 100
SPECTRO_TIME_RES_SEG = SPECTRO_DURATION_S/SPECTRO_SIZE_SEG[1]
SEGMENTATION_GAUSSIAN_SCALE = 10 / SPECTRO_TIME_RES_SEG

def load_spectro_for_seg(input_data):
    file_path = input_data[0]
    events = input_data[1:]
    nb_events = int(len(events))

    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)  # grayscale
    img = tf.image.convert_image_dtype(img, tf.uint8)  # 0-255 pixels
    img = tf.image.resize(img, size=SPECTRO_SIZE_SEG)  # resize to standard size

    if nb_events == 0:
        # if the sample is negative we consider it contains no signal of interest
        return img, tf.zeros(SPECTRO_SIZE_SEG[1])

    # we have a positive sample, we extract the location of the signals of interest in it
    events_array = tf.TensorArray(tf.float32, size=nb_events, clear_after_read=False)
    for e in range(nb_events):
        event = events[e]
        # now get the idx of the x-axis of the image where the event occurs
        events_array = events_array.write(e, (SPECTRO_DURATION_S/2 + tf.strings.to_number(event)) / SPECTRO_TIME_RES_SEG)
    events = events_array.stack()

    distrib = tfp.distributions.Normal(loc=0, scale=SEGMENTATION_GAUSSIAN_SCALE)
    y_list = []  # list of tensors that each represent 1 time step, will be concatenated as function output
    for i in range(SPECTRO_SIZE_SEG[1]):
        d = tf.math.reduce_min(tf.abs(float(i) - events)) # "distance" in pixels separating this time step from the closest event
        y_list.append(distrib.prob(d) / distrib.prob(0))  # assigned ground truth of this time step, normalized to [0,1]

    y = tf.stack(y_list)  # obtain the tensor containing all assigned ground truth values
    return img, y

def list_to_generator_seg(csv_lines):
    while True:
        for line in csv_lines:
            res = [line[0]]
            res.extend(line[2:])
            yield tuple(res)

def load_spectro_for_class(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)  # RGB
    img = tf.image.convert_image_dtype(img, tf.uint8)  # 0-255 pixels
    img = tf.image.resize(img, size=(224, 224))  # resize to standard resnet input size

    if tf.strings.regex_full_match(file_path, ".*negative.*"):
        return img, 0
    else:
        return img, 1

