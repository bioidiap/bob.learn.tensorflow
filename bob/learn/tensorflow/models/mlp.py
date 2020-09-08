import tensorflow as tf


class MLP(tf.keras.Model):
    """An MLP that can be trained with center loss and cross entropy."""

    def __init__(
        self,
        n_classes=1,
        hidden_layers=(256, 128, 64, 32),
        weight_decay=1e-5,
        name="MLP",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        dense_kw = {}
        if weight_decay is not None:
            dense_kw["kernel_regularizer"] = tf.keras.regularizers.l2(weight_decay)

        sequential_layers = []
        for i, n in enumerate(hidden_layers, start=1):
            sequential_layers.extend(
                [
                    tf.keras.layers.Dense(
                        n, use_bias=False, name=f"dense_{i}", **dense_kw
                    ),
                    tf.keras.layers.BatchNormalization(scale=False, name=f"bn_{i}"),
                    tf.keras.layers.Activation("relu", name=f"relu_{i}"),
                ]
            )

        sequential_layers.append(
            tf.keras.layers.Dense(n_classes, name="logits", **dense_kw)
        )

        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.sequential_layers = sequential_layers
        self.prelogits_shape = hidden_layers[-1]

    def call(self, x, training=None):
        assert hasattr(
            x, "_keras_history"
        ), "The input must be wrapped inside a keras Input layer."

        for i, layer in enumerate(self.sequential_layers):
            try:
                x = layer(x, training=training)
            except TypeError:
                x = layer(x)

        return x

    @property
    def prelogits(self):
        return self.layers[-2].output


class MLPDropout(tf.keras.Model):
    """An MLP that can be trained with center loss and cross entropy."""

    def __init__(
        self,
        n_classes=1,
        hidden_layers=(256, 128, 64, 32),
        weight_decay=1e-5,
        drop_rate=0.5,
        name="MLP",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        dense_kw = {}
        if weight_decay is not None:
            dense_kw["kernel_regularizer"] = tf.keras.regularizers.l2(weight_decay)

        sequential_layers = []
        for i, n in enumerate(hidden_layers, start=1):
            sequential_layers.extend(
                [
                    tf.keras.layers.Dense(
                        n, use_bias=False, name=f"dense_{i}", **dense_kw
                    ),
                    tf.keras.layers.Activation("relu", name=f"relu_{i}"),
                    tf.keras.layers.Dropout(rate=drop_rate, name=f"drop_{i}"),
                ]
            )

        sequential_layers.append(
            tf.keras.layers.Dense(n_classes, name="logits", **dense_kw)
        )

        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.drop_rate = drop_rate
        self.sequential_layers = sequential_layers
        self.prelogits_shape = hidden_layers[-1]

    def call(self, x, training=None):
        assert hasattr(
            x, "_keras_history"
        ), "The input must be wrapped inside a keras Input layer."

        for i, layer in enumerate(self.sequential_layers):
            try:
                x = layer(x, training=training)
            except TypeError:
                x = layer(x)

        return x

    @property
    def prelogits(self):
        return self.layers[-2].output
