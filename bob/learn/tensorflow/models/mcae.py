"""Multi-channel autoencoder network used in:
@inproceedings{NikisinsICB2019,
    author = {Olegs Nikisins, Anjith George, Sebastien Marcel},
    title = {Domain Adaptation in Multi-Channel Autoencoder based Features for Robust Face Anti-Spoofing},
    year = {2019},
    booktitle = {ICB 2019},
}
"""
import tensorflow as tf


def get_l2_kw(weight_decay):
    l2_kw = {}
    if weight_decay is not None:
        l2_kw = {"kernel_regularizer": tf.keras.regularizers.l2(weight_decay)}
    return l2_kw


class ConvEncoder(tf.keras.Model):
    """The encoder part"""

    def __init__(
        self,
        encoder_layers,
        data_format="channels_last",
        weight_decay=1e-5,
        name="Encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.data_format = data_format
        l2_kw = get_l2_kw(weight_decay)
        layers = []
        for i, (filters, kernel_size, strides, padding) in enumerate(encoder_layers):
            pad = tf.keras.layers.ZeroPadding2D(
                padding=padding, data_format=data_format, name=f"pad_{i}"
            )
            conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size,
                strides,
                data_format=data_format,
                name=f"conv_{i}",
                **l2_kw,
            )
            act = tf.keras.layers.Activation("relu", name=f"relu_{i}")
            pool = tf.keras.layers.MaxPooling2D(
                data_format=data_format, name=f"pool_{i}"
            )
            layers.extend([pad, conv, act, pool])
        self.sequential_layers = layers

    def call(self, x, training=None):
        for layer in self.sequential_layers:
            x = layer(x)
        return x


class ConvDecoder(tf.keras.Model):
    """The encoder part"""

    def __init__(
        self,
        decoder_layers,
        data_format="channels_last",
        weight_decay=1e-5,
        name="Decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.data_format = data_format
        l2_kw = get_l2_kw(weight_decay)
        layers = []
        for i, (filters, kernel_size, strides, cropping) in enumerate(decoder_layers):
            dconv = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                data_format=data_format,
                name=f"dconv_{i}",
                **l2_kw,
            )
            crop = tf.keras.layers.Cropping2D(
                cropping=cropping, data_format=data_format, name=f"crop_{i}"
            )
            if i == len(decoder_layers) - 1:
                act = tf.keras.layers.Activation("tanh", name="tanh")
            else:
                act = tf.keras.layers.Activation("relu", name=f"relu_{i}")
            layers.extend([dconv, crop, act])
        self.sequential_layers = layers

    def call(self, x, training=None):
        for layer in self.sequential_layers:
            x = layer(x)
        return x


class ConvAutoencoder(tf.keras.Model):
    """
    A class defining a simple convolutional autoencoder.

    Multi-channel autoencoder network used in::

        @inproceedings{NikisinsICB2019,
            author = {Olegs Nikisins, Anjith George, Sebastien Marcel},
            title = {Domain Adaptation in Multi-Channel Autoencoder based Features for Robust Face Anti-Spoofing},
            year = {2019},
            booktitle = {ICB 2019},
        }

    Attributes
    ----------
    data_format : str
        Either channels_last or channels_first
    decoder : object
        The encoder part
    encoder : object
        The decoder part
    """

    def __init__(
        self,
        data_format="channels_last",
        encoder_layers=((16, 5, 1, 2), (16, 5, 1, 2), (16, 3, 1, 2), (16, 3, 1, 2)),
        decoder_layers=(
            (16, 3, 2, 1),
            (16, 3, 2, 1),
            (16, 5, 2, 2),
            (3, 5, 2, 2),
            (3, 2, 1, 1),
        ),
        weight_decay=1e-5,
        name="ConvAutoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.data_format = data_format
        self.weight_decay = weight_decay

        self.encoder = ConvEncoder(
            encoder_layers,
            data_format=data_format,
            weight_decay=weight_decay,
            name="Encoder",
        )
        self.decoder = ConvDecoder(
            decoder_layers,
            data_format=data_format,
            weight_decay=weight_decay,
            name="Decoder",
        )

    def call(self, x, training=None):
        x = self.encoder(x, training=training)
        self.encoder_output = x
        x = self.decoder(x, training=training)
        return x
