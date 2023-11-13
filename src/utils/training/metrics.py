import tensorflow as tf

def accuracy_for_segmenter(y_true, y_pred):
    y_true = tf.math.round(tf.math.reduce_max(y_true, axis=-1))
    y_pred = tf.math.round(tf.math.reduce_max(y_pred, axis=-1))
    equality = tf.equal(y_true, y_pred)
    return tf.cast(equality, tf.keras.backend.floatx())