"""
The network using keras (same as new_architecture function below)::

    from tensorflow.python.keras import *
    from tensorflow.python.keras.layers import *
    simplecnn = Sequential([
        Conv2D(32,(3,3),padding='same',use_bias=False, input_shape=(28,28,3)),
        BatchNormalization(scale=False),
        Activation('relu'),
        MaxPool2D(padding='same'),
        Conv2D(64,(3,3),padding='same',use_bias=False),
        BatchNormalization(scale=False),
        Activation('relu'),
        MaxPool2D(padding='same'),
        Flatten(),
        Dense(1024, use_bias=False),
        BatchNormalization(scale=False),
        Activation('relu'),
        Dropout(rate=0.4),
        Dense(2, activation="softmax"),
    ])
    simplecnn.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_1 (Conv2D)            (None, 28, 28, 32)        864
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 28, 28, 32)        96
    _________________________________________________________________
    activation_1 (Activation)    (None, 28, 28, 32)        0
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 14, 14, 64)        18432
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 14, 14, 64)        192
    _________________________________________________________________
    activation_2 (Activation)    (None, 14, 14, 64)        0
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3136)              0
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              3211264
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 1024)              3072
    _________________________________________________________________
    activation_3 (Activation)    (None, 1024)              0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1024)              0
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 2050
    =================================================================
    Total params: 3,235,970
    Trainable params: 3,233,730
    Non-trainable params: 2,240
    _________________________________________________________________
"""

from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
)


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
    import pkg_resources
    from tabulate import tabulate
    from bob.learn.tensorflow.utils import model_summary

    model = SimpleCNN()
    model.summary()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
