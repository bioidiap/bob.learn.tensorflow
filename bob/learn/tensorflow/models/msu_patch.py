"""Patch-based CNN used for face PAD in:
Y. Atoum, Y. Liu, A. Jourabloo, and X. Liu, “Face anti-spoofing using patch and
depth-based CNNs,” in 2017 IEEE International Joint Conference on Biometrics (IJCB),
Denver, CO, 2017, pp. 319–328.
"""


import tensorflow as tf


def MSUPatch(name="MSUPatch", **kwargs):

    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                50,
                (5, 5),
                padding="same",
                use_bias=False,
                name="Conv-1",
                input_shape=(96, 96, 3),
            ),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-1"),
            tf.keras.layers.Activation("relu", name="ReLU-1"),
            tf.keras.layers.MaxPool2D(padding="same", name="MaxPool-1"),
            tf.keras.layers.Conv2D(
                100, (3, 3), padding="same", use_bias=False, name="Conv-2"
            ),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-2"),
            tf.keras.layers.Activation("relu", name="ReLU-2"),
            tf.keras.layers.MaxPool2D(padding="same", name="MaxPool-2"),
            tf.keras.layers.Conv2D(
                150, (3, 3), padding="same", use_bias=False, name="Conv-3"
            ),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-3"),
            tf.keras.layers.Activation("relu", name="ReLU-3"),
            tf.keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding="same", name="MaxPool-3"
            ),
            tf.keras.layers.Conv2D(
                200, (3, 3), padding="same", use_bias=False, name="Conv-4"
            ),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-4"),
            tf.keras.layers.Activation("relu", name="ReLU-4"),
            tf.keras.layers.MaxPool2D(padding="same", name="MaxPool-4"),
            tf.keras.layers.Conv2D(
                250, (3, 3), padding="same", use_bias=False, name="Conv-5"
            ),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-5"),
            tf.keras.layers.Activation("relu", name="ReLU-5"),
            tf.keras.layers.MaxPool2D(padding="same", name="MaxPool-5"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(1000, use_bias=False, name="FC-1"),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-6"),
            tf.keras.layers.Activation("relu", name="ReLU-6"),
            tf.keras.layers.Dropout(rate=0.5, name="Dropout"),
            tf.keras.layers.Dense(400, use_bias=False, name="FC-2"),
            tf.keras.layers.BatchNormalization(scale=False, name="BN-7"),
            tf.keras.layers.Activation("relu", name="ReLU-7"),
            tf.keras.layers.Dense(2, name="FC-3"),
        ],
        name=name,
        **kwargs
    )


if __name__ == "__main__":
    import pkg_resources  # noqa: F401
    from tabulate import tabulate

    from bob.learn.tensorflow.utils import model_summary

    model = MSUPatch()
    model.summary()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
