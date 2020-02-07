import tensorflow as tf
from .densenet import densenet161, ConvBlock


def _get_l2_kw(weight_decay):
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
        l2_kw = _get_l2_kw(weight_decay)
        layers = []
        for i, (filters, kernel_size, strides, padding) in enumerate(encoder_layers):
            bn = tf.keras.layers.BatchNormalization(
                scale=False, fused=False, name=f"bn_{i}"
            )
            if i == 0:
                act = tf.keras.layers.Activation("linear", name=f"linear_{i}")
            else:
                act = tf.keras.layers.Activation("relu", name=f"relu_{i}")
            pad = tf.keras.layers.ZeroPadding2D(
                padding=padding, data_format=data_format, name=f"pad_{i}"
            )
            conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                use_bias=(i == len(encoder_layers) - 1),
                data_format=data_format,
                name=f"conv_{i}",
                **l2_kw,
            )
            if i == len(encoder_layers) - 1:
                pool = tf.keras.layers.AvgPool2D(
                    data_format=data_format, name=f"pool_{i}"
                )
            else:
                pool = tf.keras.layers.MaxPooling2D(
                    data_format=data_format, name=f"pool_{i}"
                )
            layers.extend([bn, act, pad, conv, pool])
        self.sequential_layers = layers

    def call(self, x, training=None):
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


class ConvDecoder(tf.keras.Model):
    """The encoder part"""

    def __init__(
        self, decoder_layers, y_dim, weight_decay=1e-5, name="Decoder", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.data_format = data_format = "channels_last"
        self.y_dim = y_dim
        l2_kw = _get_l2_kw(weight_decay)
        layers = []
        for i, (filters, kernel_size, strides, cropping) in enumerate(decoder_layers):
            dconv = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                use_bias=False,
                data_format=data_format,
                name=f"dconv_{i}",
                **l2_kw,
            )
            crop = tf.keras.layers.Cropping2D(
                cropping=cropping, data_format=data_format, name=f"crop_{i}"
            )
            bn = tf.keras.layers.BatchNormalization(
                scale=(i == len(decoder_layers) - 1), fused=False, name=f"bn_{i}"
            )
            if i == len(decoder_layers) - 1:
                act = tf.keras.layers.Activation("tanh", name=f"tanh_{i}")
            else:
                act = tf.keras.layers.Activation("relu", name=f"relu_{i}")
            layers.extend([dconv, crop, bn, act])
        self.sequential_layers = layers

    def call(self, x, y, training=None):
        y = tf.reshape(tf.cast(y, x.dtype), (-1, 1, 1, self.y_dim))
        x = tf.concat([x, y], axis=-1)
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


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

    def __init__(
        self, encoder, decoder, z_dim, weight_decay=1e-5, name="Autoencoder", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        data_format = "channels_last"
        self.data_format = data_format
        self.weight_decay = weight_decay
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = z_dim

    def call(self, x, y, training=None):
        self.encoder_output = tf.reshape(
            self.encoder(x, training=training), (-1, 1, 1, self.z_dim)
        )
        self.decoder_output = self.decoder(self.encoder_output, y, training=training)
        return self.decoder_output


def densenet161_autoencoder(z_dim=256, y_dim=3, weight_decay=1e-10):

    encoder = densenet161(output_classes=z_dim, weight_decay=weight_decay, weights=None)
    decoder_layers = (
        (128, 7, 7, 0),
        (64, 4, 2, 1),
        (32, 4, 2, 1),
        (16, 4, 2, 1),
        (8, 4, 2, 1),
        (4, 4, 2, 1),
        (3, 1, 1, 0),
    )
    decoder = ConvDecoder(
        decoder_layers, y_dim=y_dim, weight_decay=weight_decay, name="Decoder"
    )
    autoencoder = Autoencoder(encoder, decoder, z_dim=z_dim, weight_decay=weight_decay)
    return autoencoder


class ConvDecoderSupervised(tf.keras.Model):
    """The encoder part"""

    def __init__(
        self,
        decoder_layers,
        weight_decay=1e-5,
        data_format="channels_last",
        name="Decoder",
        y_dim=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.data_format = data_format
        self.y_dim = y_dim
        l2_kw = _get_l2_kw(weight_decay)
        layers = []
        for i, (filters, kernel_size, strides, cropping) in enumerate(decoder_layers):
            dconv = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                use_bias=False,
                data_format=data_format,
                name=f"dconv_{i}",
                **l2_kw,
            )
            crop = tf.keras.layers.Cropping2D(
                cropping=cropping, data_format=data_format, name=f"crop_{i}"
            )
            bn = tf.keras.layers.BatchNormalization(
                scale=(i == len(decoder_layers) - 1), fused=False, name=f"bn_{i}"
            )
            if i == len(decoder_layers) - 1:
                act = tf.keras.layers.Activation("tanh", name=f"tanh_{i}")
            else:
                act = tf.keras.layers.Activation("relu", name=f"relu_{i}")
            layers.extend([dconv, crop, bn, act])
        self.sequential_layers = layers

    def call(self, x, training=None):
        x = tf.reshape(x, (-1, 1, 1, x.get_shape().as_list()[-1]))
        if self.y_dim is not None:
            y_fixed = tf.one_hot([[[0]]], self.y_dim, dtype=x.dtype)
            y_fixed = tf.tile(y_fixed, multiples=[tf.shape(x)[0], 1, 1, 1])
            x = tf.concat([x, y_fixed], axis=-1)
        x = tf.keras.Input(tensor=x)
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


def densenet161_autoencoder_supervised(
    x,
    training,
    weight_decay=1e-10,
    z_dim=256,
    y_dim=1,
    deeppixbis_add_one_more_layer=False,
    start_from_face_autoencoder=False,
):
    data_format = "channels_last"
    with tf.name_scope("Autoencoder"):
        densenet = densenet161(
            output_classes=z_dim,
            weight_decay=weight_decay,
            weights=None,
            data_format=data_format,
        )
        z = densenet(x, training=training)
        transition = tf.keras.Input(tensor=densenet.transition_blocks[1].output)

        layers = [
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                data_format=data_format,
                name="dec",
            ),
            tf.keras.layers.Flatten(
                data_format=data_format, name="Pixel_Logits_Flatten"
            ),
        ]

        if deeppixbis_add_one_more_layer:
            layers.insert(
                0,
                ConvBlock(
                    num_filters=32,
                    data_format=data_format,
                    bottleneck=True,
                    weight_decay=weight_decay,
                    name="prelogits",
                ),
            )

        y = transition
        with tf.name_scope("DeepPixBiS"):
            for l in layers:
                try:
                    y = l(y, training=training)
                except TypeError:
                    y = l(y)

        deep_pix_bis_final_layers = tf.keras.Model(
            inputs=transition, outputs=y, name="DeepPixBiS"
        )
        encoder = tf.keras.Model(inputs=[x, transition], outputs=[y, z], name="Encoder")
        encoder.densenet = densenet
        if deeppixbis_add_one_more_layer:
            encoder.prelogits = deep_pix_bis_final_layers.layers[-3].output
        else:
            encoder.prelogits = transition
        encoder.deep_pix_bis = deep_pix_bis_final_layers
        decoder_layers = (
            (128, 7, 7, 0),
            (64, 4, 2, 1),
            (32, 4, 2, 1),
            (16, 4, 2, 1),
            (8, 4, 2, 1),
            (4, 4, 2, 1),
            (3, 1, 1, 0),
        )
        decoder = ConvDecoderSupervised(
            decoder_layers,
            weight_decay=weight_decay,
            name="Decoder",
            data_format=data_format,
            y_dim=3 if start_from_face_autoencoder else None,
        )
        x_hat = decoder(z, training=training)
        autoencoder = tf.keras.Model(
            inputs=[x, transition], outputs=[y, z, x_hat], name="Autoencoder"
        )
        autoencoder.encoder = encoder
        autoencoder.decoder = decoder
    return autoencoder, y, z, x_hat
