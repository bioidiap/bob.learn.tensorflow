import tensorflow as tf

from ..dataset import tf_repeat
from .utils import balanced_sigmoid_cross_entropy_loss_weights
from .utils import balanced_softmax_cross_entropy_loss_weights


class PixelWise:
    """A pixel wise loss which is just a cross entropy loss but applied to all pixels"""

    def __init__(
        self, balance_weights=True, n_one_hot_labels=None, label_smoothing=0.5, **kwargs
    ):
        super(PixelWise, self).__init__(**kwargs)
        self.balance_weights = balance_weights
        self.n_one_hot_labels = n_one_hot_labels
        self.label_smoothing = label_smoothing

    def __call__(self, labels, logits):
        with tf.compat.v1.name_scope("PixelWiseLoss"):
            flatten = tf.keras.layers.Flatten()
            logits = flatten(logits)
            n_pixels = logits.get_shape()[-1]
            weights = 1.0
            if self.balance_weights and self.n_one_hot_labels:
                # use labels to figure out the required loss
                weights = balanced_softmax_cross_entropy_loss_weights(
                    labels, dtype=logits.dtype
                )
                # repeat weights for all pixels
                weights = tf_repeat(weights[:, None], [1, n_pixels])
                weights = tf.reshape(weights, (-1,))
            elif self.balance_weights and not self.n_one_hot_labels:
                # use labels to figure out the required loss
                weights = balanced_sigmoid_cross_entropy_loss_weights(
                    labels, dtype=logits.dtype
                )
                # repeat weights for all pixels
                weights = tf_repeat(weights[:, None], [1, n_pixels])

            if self.n_one_hot_labels:
                labels = tf_repeat(labels, [n_pixels, 1])
                labels = tf.reshape(labels, (-1, self.n_one_hot_labels))
                # reshape logits too as softmax_cross_entropy is buggy and cannot really
                # handle higher dimensions
                logits = tf.reshape(logits, (-1, self.n_one_hot_labels))
                loss_fn = tf.compat.v1.losses.softmax_cross_entropy
            else:
                labels = tf.reshape(labels, (-1, 1))
                labels = tf_repeat(labels, [n_pixels, 1])
                labels = tf.reshape(labels, (-1, n_pixels))
                loss_fn = tf.compat.v1.losses.sigmoid_cross_entropy

            loss_pixel_wise = loss_fn(
                labels,
                logits=logits,
                weights=weights,
                label_smoothing=self.label_smoothing,
                reduction=tf.compat.v1.losses.Reduction.MEAN,
            )
        tf.compat.v1.summary.scalar("loss_pixel_wise", loss_pixel_wise)
        return loss_pixel_wise
