import tensorflow as tf
from ..utils import to_channels_first, to_channels_last


def fixed_image_standardization(image):
    # normalize uint8 image between -0.5 and 0.5
    return tf.cast(image, tf.float32) / 255.0 - 0.5
