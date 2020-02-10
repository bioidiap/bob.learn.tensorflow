"""Face Autoencoder used in:
IMPROVING CROSS-DATASET PERFORMANCE OF FACE PRESENTATION ATTACK DETECTION SYSTEMS USING FACE RECOGNITION DATASETS,
Mohammadi, Amir and Bhattacharjee, Sushil and Marcel, Sebastien, ICASSP 2020
"""

import tensorflow as tf
from bob.learn.tensorflow.models.densenet import densenet161


def _get_l2_kw(weight_decay):
    l2_kw = {}
    if weight_decay is not None:
        l2_kw = {"kernel_regularizer": tf.keras.regularizers.l2(weight_decay)}
    return l2_kw


def ConvDecoder(
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
    last_act="tanh",
    name="Decoder",
    **kwargs,
):
    """The decoder similar to the one in
    https://github.com/google/compare_gan/blob/master/compare_gan/architectures/sndcgan.py
    """
    z_dim = z_dim
    data_format = "channels_last"
    l2_kw = _get_l2_kw(weight_decay)
    layers = [
        tf.keras.layers.Reshape(
            (1, 1, z_dim), input_shape=(z_dim,), name=f"{name}/reshape"
        )
    ]
    for i, (filters, kernel_size, strides, cropping) in enumerate(decoder_layers):
        dconv = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            use_bias=i == len(decoder_layers) - 1,
            data_format=data_format,
            name=f"{name}/dconv_{i}",
            **l2_kw,
        )
        crop = tf.keras.layers.Cropping2D(
            cropping=cropping, data_format=data_format, name=f"{name}/crop_{i}"
        )

        if i == len(decoder_layers) - 1:
            act = tf.keras.layers.Activation(
                f"{last_act}", name=f"{name}/{last_act}_{i}"
            )
            bn = None
        else:
            act = tf.keras.layers.Activation("relu", name=f"{name}/relu_{i}")
            bn = tf.keras.layers.BatchNormalization(
                scale=False, fused=False, name=f"{name}/bn_{i}"
            )
        if bn is not None:
            layers.extend([dconv, crop, bn, act])
        else:
            layers.extend([dconv, crop, act])
    return tf.keras.Sequential(layers, name=name, **kwargs)


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


def autoencoder_face(z_dim=256, weight_decay=1e-10, decoder_last_act="tanh"):
    encoder = densenet161(
        output_classes=z_dim, weight_decay=weight_decay, weights=None, name="DenseNet"
    )
    decoder = ConvDecoder(
        z_dim=z_dim,
        weight_decay=weight_decay,
        last_act=decoder_last_act,
        name="Decoder",
    )
    autoencoder = Autoencoder(encoder, decoder, name="Autoencoder")
    return autoencoder


if __name__ == "__main__":
    import pkg_resources
    from tabulate import tabulate
    from bob.learn.tensorflow.utils import model_summary

    model = ConvDecoder(z_dim=256, weight_decay=1e-9, last_act="tanh", name="Decoder")
    model.summary()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
