import tensorflow as tf


class ConvDiscriminator(tf.keras.Model):
    """A discriminator that can sit on top of DenseNet 161's transition 1 block.
    The output of that block given 224x224 inputs is 14x14x384."""

    def __init__(self, data_format="channels_last", n_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.n_classes = n_classes
        act = "sigmoid" if n_classes == 1 else "softmax"
        self.sequential_layers = [
            tf.keras.layers.Conv2D(200, 1, data_format=data_format),
            tf.keras.layers.Activation("relu"),
            tf.layers.AveragePooling2D(3, 2, data_format=data_format),
            tf.keras.layers.Conv2D(100, 1, data_format=data_format),
            tf.keras.layers.Activation("relu"),
            tf.layers.AveragePooling2D(3, 2, data_format=data_format),
            tf.keras.layers.Flatten(data_format=data_format),
            tf.keras.layers.Dense(n_classes),
            tf.keras.layers.Activation(act),
        ]

    def call(self, x, training=None):
        for l in self.sequential_layers:
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
            x = l(x)
        return x
