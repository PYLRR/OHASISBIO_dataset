import tensorflow as tf

def segmenter_res_to_classifier(y_true, y_pred):
    y_true = tf.math.reduce_max(y_true, axis=-1)
    y_pred = tf.math.reduce_max(y_pred, axis=-1)
    return y_true, y_pred

def accuracy_for_segmenter(y_true, y_pred):
    y_true, y_pred = segmenter_res_to_classifier(y_true, y_pred)
    equality = tf.equal(tf.math.round(y_true), tf.math.round(y_pred))
    return tf.cast(equality, tf.keras.backend.floatx())

class AUC_for_segmenter(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = segmenter_res_to_classifier(y_true, y_pred)
        return super().update_state(y_true, y_pred)