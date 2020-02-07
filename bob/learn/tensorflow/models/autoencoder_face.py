import tensorflow as tf
from .densenet import densenet161


def _get_l2_kw(weight_decay):
    l2_kw = {}
    if weight_decay is not None:
        l2_kw = {"kernel_regularizer": tf.keras.regularizers.l2(weight_decay)}
    return l2_kw


class ConvDecoder(tf.keras.Sequential):
    """The decoder similar to the one in
    https://github.com/google/compare_gan/blob/master/compare_gan/architectures/sndcgan.py
    """

    def __init__(
        self,
        z_dim,
        decoder_layers=(
            (512, 7, 7, 0),
            (256, 4, 2, 1),
            (128, 4, 2, 1),
            (64, 4, 2, 1),
            (32, 4, 2, 1),
            (16, 4, 2, 1),
            (3, 1, 1, 0),
        ),
        weight_decay=1e-5,
        name="Decoder",
        **kwargs,
    ):
        self.z_dim = z_dim
        self.data_format = data_format = "channels_last"
        l2_kw = _get_l2_kw(weight_decay)
        layers = [
            tf.keras.layers.Reshape((1, 1, z_dim), input_shape=(z_dim,), name="reshape")
        ]
        for i, (filters, kernel_size, strides, cropping) in enumerate(decoder_layers):
            dconv = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                use_bias=i == len(decoder_layers) - 1,
                data_format=data_format,
                name=f"dconv_{i}",
                **l2_kw,
            )
            crop = tf.keras.layers.Cropping2D(
                cropping=cropping, data_format=data_format, name=f"crop_{i}"
            )

            if i == len(decoder_layers) - 1:
                act = tf.keras.layers.Activation("tanh", name=f"tanh_{i}")
                bn = None
            else:
                act = tf.keras.layers.Activation("relu", name=f"relu_{i}")
                bn = tf.keras.layers.BatchNormalization(
                    scale=False, fused=False, name=f"bn_{i}"
                )
            if bn is not None:
                layers.extend([dconv, crop, bn, act])
            else:
                layers.extend([dconv, crop, act])
        with tf.name_scope(name):
            super().__init__(layers=layers, name=name, **kwargs)


class Autoencoder(tf.keras.Model):
    """
    A class defining a simple convolutional autoencoder.

    Attributes
    ----------
    data_format : str
        channels_last is only supported
    decoder : object
        The encoder part
    encoder : object
        The decoder part
    """

    def __init__(self, encoder, decoder, name="Autoencoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x, training=None):
        z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        return z, x_hat

def autoencoder_face(z_dim=256, weight_decay=1e-9):
    encoder = densenet161(
        output_classes=z_dim, weight_decay=weight_decay, weights=None, name="DenseNet"
    )
    decoder = ConvDecoder(z_dim=z_dim, weight_decay=weight_decay, name="Decoder")
    autoencoder = Autoencoder(encoder, decoder, name="Autoencoder")
    return autoencoder
