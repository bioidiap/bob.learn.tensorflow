import tensorflow as tf
from ..gan.spectral_normalization import spectral_norm_regularizer
from ..utils import gram_matrix


class ConvDiscriminator(tf.keras.Model):
    """A discriminator that can sit on top of DenseNet 161's transition 1 block.
    The output of that block given 224x224x3 inputs is 14x14x384."""

    def __init__(self, data_format="channels_last", n_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.n_classes = n_classes
        act = "sigmoid" if n_classes == 1 else "softmax"
        self.sequential_layers = [
            tf.keras.layers.Conv2D(200, 1, data_format=data_format),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.AveragePooling2D(3, 2, data_format=data_format),
            tf.keras.layers.Conv2D(100, 1, data_format=data_format),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.AveragePooling2D(3, 2, data_format=data_format),
            tf.keras.layers.Flatten(data_format=data_format),
            tf.keras.layers.Dense(n_classes),
            tf.keras.layers.Activation(act),
        ]

    def call(self, x, training=None):
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


class ConvDiscriminator2(tf.keras.Model):
    """A discriminator that can sit on top of DenseNet 161's transition 1 block.
    The output of that block given 224x224 inputs is 14x14x384. Here we want to output
    15x15x128 features which is going to match the output of encoder in mcae.py given
    these layers::

        ENCODER_LAYERS = (
        (32, 5, 1, 2),
        (64, 5, 1, 2),
        (128, 3, 1, 2),
        (128, 3, 1, 2)
        )
        DECODER_LAYERS = (
            (64, 3, 2, 1),
            (32, 3, 2, 1),
            (16, 5, 2, 2),
            (8, 5, 2, 2),
            (3, 2, 1, 1),
        )
    """

    def __init__(self, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.sequential_layers = [
            tf.keras.layers.ZeroPadding2D(
                padding=((1, 0), (1, 0)), data_format=data_format
            ),
            tf.keras.layers.Conv2D(256, 5, data_format=data_format, padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(128, 5, data_format=data_format, padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(128, 1, data_format=data_format, padding="same"),
            tf.keras.layers.Activation("relu"),
        ]

    def call(self, x, training=None):
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


class ConvDiscriminator3(tf.keras.Model):
    """A discriminator that takes images and tries its best.
    Be careful, this one returns logits."""

    def __init__(self, data_format="channels_last", n_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.n_classes = n_classes
        spectral_norm = spectral_norm_regularizer(scale=1.0)
        conv2d_kw = {"kernel_regularizer": spectral_norm, "data_format": data_format}
        self.sequential_layers = [
            tf.keras.layers.Conv2D(64, 3, strides=1, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(64, 4, strides=2, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(128, 3, strides=1, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(128, 4, strides=2, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(256, 3, strides=1, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(256, 4, strides=2, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(512, 3, strides=1, **conv2d_kw),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.GlobalAveragePooling2D(data_format=data_format),
            tf.keras.layers.Dense(n_classes),
        ]

    def call(self, x, training=None):
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


class DenseDiscriminator(tf.keras.Model):
    """A discriminator that takes vectors as input and tries its best.
    Be careful, this one returns logits."""

    def __init__(self, n_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.sequential_layers = [
            tf.keras.layers.Dense(1000),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(n_classes),
        ]

    def call(self, x, training=None):
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


class GramComparer1(tf.keras.Model):
    """A model to compare images based on their gram matrices."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.conv2d = tf.keras.layers.Conv2D(128, 7)

    def call(self, x_1_2, training=None):
        def _call(x):
            x = self.batchnorm(x, training=training)
            x = self.conv2d(x)
            return gram_matrix(x)

        gram1 = _call(x_1_2[..., :3])
        gram2 = _call(x_1_2[..., 3:])
        return -tf.reduce_mean((gram1 - gram2) ** 2, axis=[1, 2])[:, None]
