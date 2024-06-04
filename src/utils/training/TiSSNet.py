import tensorflow as tf
import tensorflow.keras.layers as layers


def TiSSNet(input=(None, None)):
    """ Create the TiSSNet model.
    :param input: Size of the input as (height, width), with None values if unknown.
    :return: The TiSSNet model.
    """

    inputs = layers.Input(shape=(input[0], input[1], 1))
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv1")(x)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv2")(x)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv3")(x)
    x = layers.MaxPooling2D((2, 1), padding='same')(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU', name="block2_conv1")(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU', name="block2_conv2")(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU', name="block2_conv3")(x)
    x = layers.MaxPooling2D((4, 1), padding='same')(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU', name="block3_conv1")(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU', name="block3_conv2")(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU', name="block3_conv3")(x)
    x = layers.MaxPooling2D((4, 1), padding='same')(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU', name="block4_conv1")(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU', name="block4_conv2")(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU', name="block4_conv3")(x)
    x = layers.MaxPooling2D((4, 1), padding='same')(x)
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name="output_conv")(x)
    outputs = layers.Flatten()(x)

    model = tf.keras.Model(inputs, outputs, name="time_segmenter")

    return model