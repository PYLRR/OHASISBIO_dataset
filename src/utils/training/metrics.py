import tensorflow as tf


def segmenter_res_to_classifier(y_true, y_pred):
    """ Converts time segmentation results to classification results, simply taking max values along the time axis.
    :param y_true: The ground truths associated to the time segmentation results.
    :param y_pred: The values predicted by the time segmentation model.
    :return: Values in the range [0,1].
    """
    y_true = tf.math.reduce_max(y_true, axis=-1)
    y_pred = tf.math.reduce_max(y_pred, axis=-1)
    return y_true, y_pred


def accuracy_for_segmenter(y_true, y_pred):
    """ Computes the accuracy score for time segmentation models as if they were classifiers.
    :param y_true: The ground truths associated to the time segmentation results.
    :param y_pred: The values predicted by the time segmentation model.
    :return: The percentage of success of the predictions.
    """
    y_true, y_pred = segmenter_res_to_classifier(y_true, y_pred)
    equality = tf.equal(tf.math.round(y_true), tf.math.round(y_pred))
    return tf.cast(equality, tf.keras.backend.floatx())


class AUC_for_segmenter(tf.keras.metrics.AUC):
    """ Class enabling to compute the Area Under Curve (AUC) of the ROC curve for time segmentation models. """

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Computes the AUC score for time segmentation models as if they were classifiers.
        :param y_true: The ground truths associated to the time segmentation results.
        :param y_pred: The values predicted by the time segmentation model.
        :param sample_weight: The weight of these samples in the total score.
        :return: An AUC estimation for these samples.
        """
        y_true, y_pred = segmenter_res_to_classifier(y_true, y_pred)
        return super().update_state(y_true, y_pred, sample_weight)
