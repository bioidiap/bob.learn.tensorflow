import tensorflow as tf


def AlexNet_simplified(name="AlexNet", **kwargs):
    """A simplified implementation of AlexNet presented in:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to
    document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(227, 227, 3)),
            tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, name="C1", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name="P1"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, name="C2", activation="relu", padding="same"),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name="P2"),
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, name="C3", activation="relu", padding="same"),
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, name="C4", activation="relu", padding="same"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, name="C5", activation="relu", padding="same"),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name="P5"),
            tf.keras.layers.Flatten(name="FLATTEN"),
            tf.keras.layers.Dropout(rate=0.5, name="D6"),
            tf.keras.layers.Dense(units=4096, activation="relu", name="F6"),
            tf.keras.layers.Dropout(rate=0.5, name="D7"),
            tf.keras.layers.Dense(units=4096, activation="relu", name="F7"),
            tf.keras.layers.Dense(units=1000, activation="softmax", name="OUTPUT"),
        ],
        name=name,
        **kwargs
    )
    return model


if __name__ == "__main__":
    import pkg_resources
    from bob.learn.tensorflow.utils import model_summary

    model = AlexNet_simplified()
    model.summary()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    from tabulate import tabulate

    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
