import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import resnet50

def time_segmenter_model():
    inputs = layers.Input(shape=(None, None, 1))
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(16, 8, padding='same', activation='LeakyReLU')(x)
    x = layers.MaxPooling2D((2,1), padding='same')(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(32, (5, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.MaxPooling2D((4,1), padding='same')(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(64, (3, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.MaxPooling2D((4,1), padding='same')(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.Conv2D(128, (2, 8), padding='same', activation='LeakyReLU')(x)
    x = layers.MaxPooling2D((4,1), padding='same')(x)
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    outputs = layers.Flatten()(x)

    model = tf.keras.Model(inputs, outputs, name="time_segmenter")

    return model

def resnet_model():
    inputs = layers.Input(shape=(224, 224, 3))

    base = resnet50.ResNet50(weights="imagenet", include_top=False)(inputs)

    x = layers.MaxPooling2D(pool_size=(7, 7))(base)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="LeakyReLU")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_classifier")

    return model