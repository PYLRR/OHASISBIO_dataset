import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import resnet50

def time_segmenter_model(input=(None, None)):
    inputs = layers.Input(shape=(input[0], input[1], 1))
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv1")(x)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv2")(x)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU', name="block1_conv3")(x)
    x = layers.MaxPooling2D((2,1), padding='same')(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU', name="block2_conv1")(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU', name="block2_conv2")(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU', name="block2_conv3")(x)
    x = layers.MaxPooling2D((4,1), padding='same')(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU', name="block3_conv1")(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU', name="block3_conv2")(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU', name="block3_conv3")(x)
    x = layers.MaxPooling2D((4,1), padding='same')(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU', name="block4_conv1")(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU', name="block4_conv2")(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU', name="block4_conv3")(x)
    x = layers.MaxPooling2D((4,1), padding='same')(x)
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name="output_conv")(x)
    outputs = layers.Flatten()(x)

    model = tf.keras.Model(inputs, outputs, name="time_segmenter")

    return model


def get_phasenet_model(SIZE):
    def phasenet_like_model():
        inputs = layers.Input(shape=(SIZE,))

        x = layers.Reshape((SIZE, 1))(inputs)

        x = layers.Conv1D(8, 7, padding='same', activation='relu')(x)
        x1 = layers.Conv1D(8, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(8, 7, padding='same', strides=4, activation='relu')(x1)
        x2 = layers.Conv1D(11, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(11, 7, padding='same', strides=4, activation='relu')(x2)
        x3 = layers.Conv1D(16, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(16, 7, padding='same', strides=4, activation='relu')(x3)
        x4 = layers.Conv1D(22, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(22, 7, padding='same', strides=4, activation='relu')(x4)
        x5 = layers.Conv1D(29, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(29, 7, padding='same', strides=4, activation='relu')(x5)
        x6 = layers.Conv1D(37, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(37, 7, padding='same', strides=4, activation='relu')(x6)

        x = layers.Conv1D(46, 4, padding='same', activation='relu')(x)

        x = layers.Conv1DTranspose(37, 7, padding='same', strides=4, activation='relu')(x)
        x = layers.concatenate([x6, x])
        x = layers.Conv1D(37, 7, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(29, 7, padding='same', strides=4, activation='relu')(x)
        x = layers.concatenate([x5, x])
        x = layers.Conv1D(29, 7, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(22, 7, padding='same', strides=4, activation='relu')(x)
        x = layers.concatenate([x4, x])
        x = layers.Conv1D(22, 7, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(16, 7, padding='same', strides=4, activation='relu')(x)
        x = layers.concatenate([x3, x])
        x = layers.Conv1D(16, 7, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(11, 7, padding='same', strides=4, activation='relu')(x)
        x = layers.concatenate([x2, x])
        x = layers.Conv1D(11, 7, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(8, 7, padding='same', strides=4, activation='relu')(x)
        x = layers.concatenate([x1, x])
        x = layers.Conv1D(8, 7, padding='same', activation='relu')(x)
        x = layers.Conv1D(1, 7, padding='same', activation='sigmoid')(x)

        outputs = layers.Flatten()(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phasenet_like")

        return model
    return phasenet_like_model

def resnet_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    base = resnet50.ResNet50(weights="imagenet", include_top=False)
    for layer in base.layers[:-5]:
        layer.trainable = False

    base = base(x)

    x = layers.MaxPooling2D(pool_size=(7, 7))(base)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation=layers.LeakyReLU())(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation=layers.LeakyReLU())(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation=layers.LeakyReLU())(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation=layers.LeakyReLU())(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=layers.LeakyReLU())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_classifier")

    return model