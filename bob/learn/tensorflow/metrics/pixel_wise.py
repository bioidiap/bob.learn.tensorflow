import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper

from ..losses.pixel_wise import get_pixel_wise_labels


def pixel_wise_binary_accuracy(labels, logits, threshold=0.5):
    n_pixels = logits.shape[-1]
    labels = get_pixel_wise_labels(labels, n_pixels)
    return tf.keras.metrics.binary_accuracy(labels, logits, threshold=threshold)


class PixelwiseBinaryAccuracy(MeanMetricWrapper):
    """Calculates accuracy from labels and pixel-wise logits.
    The labels should not be pixel-wise themselves.
    """

    def __init__(self, threshold=0.5, name="pixel_wise_accuracy", dtype=None, **kwargs):
        super().__init__(
            pixel_wise_binary_accuracy, name, dtype=dtype, threshold=threshold, **kwargs
        )
