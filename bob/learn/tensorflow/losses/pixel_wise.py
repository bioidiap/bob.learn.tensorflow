import tensorflow as tf

from ..utils import tf_repeat
from .balanced_cross_entropy import balanced_sigmoid_cross_entropy_loss_weights


def get_pixel_wise_labels(labels, n_pixels):
    labels = tf.reshape(labels, (-1, 1))
    labels = tf_repeat(labels, [n_pixels, 1])
    labels = tf.reshape(labels, (-1, n_pixels))
    return labels


class PixelwiseBinaryCrossentropy(tf.keras.losses.Loss):
    """A pixel wise loss which is just a cross entropy loss but applied to all pixels.
    Appeared in::

        @inproceedings{GeorgeICB2019,
            author = {Anjith George, Sebastien Marcel},
            title = {Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection},
            year = {2019},
            booktitle = {ICB 2019},
        }
    """

    def __init__(
        self,
        balance_weights=True,
        label_smoothing=0.5,
        name="pixel_wise_binary_cross_entropy",
        **kwargs
    ):
        """
        Parameters
        ----------
        balance_weights : bool, optional
            Whether the loss should be balanced per samples of different classes in each batch, by default True
        label_smoothing : float, optional
            Label smoothing, by default 0.5
        """
        super().__init__(name=name, **kwargs)
        self.balance_weights = balance_weights
        self.label_smoothing = label_smoothing

    def call(self, labels, logits):

        n_pixels = logits.shape[-1]
        pixel_wise_labels = get_pixel_wise_labels(labels, n_pixels)

        # per batch weighting, different from Keras's sample weights.
        weights = 1.0
        if self.balance_weights:
            # use labels to figure out the required loss weights
            weights = balanced_sigmoid_cross_entropy_loss_weights(
                labels, dtype=logits.dtype
            )

        loss_pixel_wise = tf.keras.losses.binary_crossentropy(
            y_true=pixel_wise_labels,
            y_pred=logits,
            label_smoothing=self.label_smoothing,
            from_logits=True,
        )
        loss_pixel_wise = loss_pixel_wise * weights
        return loss_pixel_wise
