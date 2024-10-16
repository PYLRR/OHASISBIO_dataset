import tensorflow as tf
import tensorflow.keras.layers as layers
from torch import nn


def TiSSNet_tf(input=(None, None)):
    """ Create the TiSSNet model.
    :param input: Size of the input as (height, width), with None values if unknown.
    :return: The TiSSNet model.
    """

    inputs = layers.Input(shape=(input[0], input[1], 1))
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv1")(inputs)
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



class TiSSNet_torch(nn.Module):
    def __init__(self):
        super().__init__()

        modules = []

        modules.append(nn.Conv2d(1, 16, kernel_size=(8, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(16, 16, kernel_size=(8, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(16, 16, kernel_size=(8, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(2, 1)))

        modules.append(nn.Conv2d(16, 32, kernel_size=(5, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(32, 32, kernel_size=(5, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(32, 32, kernel_size=(5, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(4, 1)))

        modules.append(nn.Conv2d(32, 64, kernel_size=(3, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(64, 64, kernel_size=(3, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(64, 64, kernel_size=(3, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(4, 1)))

        modules.append(nn.Conv2d(64, 128, kernel_size=(2, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(128, 128, kernel_size=(2, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(128, 128, kernel_size=(2, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(4, 1)))

        modules.append(nn.Conv2d(128, 1, kernel_size=1))
        modules.append(nn.Sigmoid())

        modules.append(nn.Flatten(-3, -1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)