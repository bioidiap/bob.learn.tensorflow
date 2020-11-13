"""A small CNN used for patch-based Face PAD"""

from tensorflow.python.keras import Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPool2D


def SimpleCNN(input_shape=(28, 28, 3), inputs=None, name="SimpleCNN", **kwargs):

    if inputs is None:
        inputs = Input(input_shape)
    model = Sequential(
        [
            inputs,
            Conv2D(32, (3, 3), padding="same", use_bias=False),
            BatchNormalization(scale=False),
            Activation("relu"),
            MaxPool2D(padding="same"),
            Conv2D(64, (3, 3), padding="same", use_bias=False),
            BatchNormalization(scale=False),
            Activation("relu"),
            MaxPool2D(padding="same"),
            Flatten(),
            Dense(1024, use_bias=False),
            BatchNormalization(scale=False),
            Activation("relu"),
            Dropout(rate=0.4),
            Dense(2),
        ],
        name=name,
        **kwargs
    )

    return model


if __name__ == "__main__":
    import pkg_resources  # noqa: F401
    from tabulate import tabulate

    from bob.learn.tensorflow.utils import model_summary

    model = SimpleCNN()
    model.summary()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
