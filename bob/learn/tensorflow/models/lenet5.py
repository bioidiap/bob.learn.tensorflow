import tensorflow as tf


def LeNet5_simplified(name="LeNet5", **kwargs):
    """A heavily simplified implementation of LeNet-5 presented in:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to
    document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(
                filters=6, kernel_size=5, name="C1", activation="tanh"
            ),
            tf.keras.layers.AvgPool2D(pool_size=2, name="S2"),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=5, name="C3", activation="tanh"
            ),
            tf.keras.layers.AvgPool2D(pool_size=2, name="S4"),
            tf.keras.layers.Conv2D(
                filters=120, kernel_size=5, name="C5", activation="tanh"
            ),
            tf.keras.layers.Flatten(name="FLATTEN"),
            tf.keras.layers.Dense(units=84, activation="tanh", name="F6"),
            tf.keras.layers.Dense(units=10, name="OUTPUT"),
        ],
        name=name,
        **kwargs
    )
    return model


if __name__ == "__main__":
    import pkg_resources  # noqa: F401

    from bob.learn.tensorflow.utils import model_summary

    model = LeNet5_simplified()
    model.summary()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    from tabulate import tabulate

    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
