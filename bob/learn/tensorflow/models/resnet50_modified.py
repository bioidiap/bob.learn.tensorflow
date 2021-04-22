# -*- coding: utf-8 -*-
"""
The resnet50 from `tf.keras.applications.Resnet50` has a problem with the convolutional layers.
It basically add bias terms to such layers followed by batch normalizations, which is not correct

https://github.com/tensorflow/tensorflow/issues/37365

This resnet 50 implementation provides a cleaner version
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.regularizers import l2

global weight_decay
weight_decay = 1e-4


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(
        self, kernel_size, filters, stage, block, weight_decay=1e-4, name=None, **kwargs
    ):

        """Block that has no convolutianal layer as skip connection

        Parameters
        ----------
            kernel_size:
               The kernel size of middle conv layer at main path

            filters:
                list of integers, the filterss of 3 conv layer at main path
            stage:
              Current stage label, used for generating layer names

            block:
                'a','b'..., current block label, used for generating layer names

        """
        super().__init__(name=name, **kwargs)

        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_1 = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
        bn_name_1 = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce/bn"
        layers = [
            Conv2D(
                filters1,
                (1, 1),
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_1,
            )
        ]

        layers += [BatchNormalization(axis=bn_axis, name=bn_name_1)]
        layers += [Activation("relu")]

        conv_name_2 = "conv" + str(stage) + "_" + str(block) + "_3x3"
        bn_name_2 = "conv" + str(stage) + "_" + str(block) + "_3x3/bn"
        layers += [
            Conv2D(
                filters2,
                kernel_size,
                padding="same",
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_2,
            )
        ]
        layers += [BatchNormalization(axis=bn_axis, name=bn_name_2)]
        layers += [Activation("relu")]

        conv_name_3 = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
        bn_name_3 = "conv" + str(stage) + "_" + str(block) + "_1x1_increase/bn"
        layers += [
            Conv2D(
                filters3,
                (1, 1),
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_3,
            )
        ]
        layers += [BatchNormalization(axis=bn_axis, name=bn_name_3)]
        self.layers = layers

    def call(self, input_tensor, training=None):

        x = input_tensor
        for lay in self.layers:
            x = lay(x, training=training)

        x = tf.keras.layers.add([x, input_tensor])
        x = Activation("relu")(x)

        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size,
        filters,
        stage,
        block,
        strides=(2, 2),
        weight_decay=1e-4,
        name=None,
        **kwargs,
    ):
        """Block that has a conv layer AS shortcut.
        Parameters
        ----------
            kernel_size:
               The kernel size of middle conv layer at main path

            filters:
                list of integers, the filterss of 3 conv layer at main path
            stage:
              Current stage label, used for generating layer names

            block:
                'a','b'..., current block label, used for generating layer names
        """
        super().__init__(name=name, **kwargs)

        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_1 = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
        bn_name_1 = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce/bn"
        layers = [
            Conv2D(
                filters1,
                (1, 1),
                strides=strides,
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_1,
            )
        ]
        layers += [BatchNormalization(axis=bn_axis, name=bn_name_1)]
        layers += [Activation("relu")]

        conv_name_2 = "conv" + str(stage) + "_" + str(block) + "_3x3"
        bn_name_2 = "conv" + str(stage) + "_" + str(block) + "_3x3/bn"
        layers += [
            Conv2D(
                filters2,
                kernel_size,
                padding="same",
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_2,
            )
        ]
        layers += [BatchNormalization(axis=bn_axis, name=bn_name_2)]
        layers += [Activation("relu")]

        conv_name_3 = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
        bn_name_3 = "conv" + str(stage) + "_" + str(block) + "_1x1_increase/bn"
        layers += [
            Conv2D(
                filters3,
                (1, 1),
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_3,
            )
        ]
        layers += [BatchNormalization(axis=bn_axis, name=bn_name_3)]

        conv_name_4 = "conv" + str(stage) + "_" + str(block) + "_1x1_proj"
        bn_name_4 = "conv" + str(stage) + "_" + str(block) + "_1x1_proj/bn"
        shortcut = [
            Conv2D(
                filters3,
                (1, 1),
                strides=strides,
                kernel_initializer="orthogonal",
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=conv_name_4,
            )
        ]
        shortcut += [BatchNormalization(axis=bn_axis, name=bn_name_4)]

        self.layers = layers
        self.shortcut = shortcut

    def call(self, input_tensor, training=None):
        x = input_tensor
        for lay in self.layers:
            x = lay(x, training=training)

        x_s = input_tensor
        for lay in self.shortcut:
            x_s = lay(x_s, training=training)

        x = tf.keras.layers.add([x, x_s])
        x = Activation("relu")(x)
        return x


def resnet50_modified(input_tensor=None, input_shape=None, **kwargs):
    """
    The resnet50 from `tf.keras.applications.Resnet50` has a problem with the convolutional layers.
    It basically add bias terms to such layers followed by batch normalizations, which is not correct

    https://github.com/tensorflow/tensorflow/issues/37365

    This resnet 50 implementation provides a cleaner version

    """
    if input_tensor is None:
        input_tensor = tf.keras.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            input_tensor = tf.keras.Input(tensor=input_tensor, shape=input_shape)

    bn_axis = 3
    # inputs are of size 224 x 224 x 3
    layers = [input_tensor]
    layers += [
        Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            kernel_initializer="orthogonal",
            use_bias=False,
            trainable=True,
            kernel_regularizer=l2(weight_decay),
            padding="same",
            name="conv1/7x7_s2",
        )
    ]

    # inputs are of size 112 x 112 x 64
    layers += [BatchNormalization(axis=bn_axis, name="conv1/7x7_s2/bn")]
    layers += [Activation("relu")]
    layers += [MaxPooling2D((3, 3), strides=(2, 2))]

    # inputs are of size 56 x 56
    layers += [ConvBlock(3, [64, 64, 256], stage=2, block=1, strides=(1, 1))]
    layers += [IdentityBlock(3, [64, 64, 256], stage=2, block=2)]
    layers += [IdentityBlock(3, [64, 64, 256], stage=2, block=3)]

    # inputs are of size 28 x 28
    layers += [ConvBlock(3, [128, 128, 512], stage=3, block=1)]
    layers += [IdentityBlock(3, [128, 128, 512], stage=3, block=2)]
    layers += [IdentityBlock(3, [128, 128, 512], stage=3, block=3)]
    layers += [IdentityBlock(3, [128, 128, 512], stage=3, block=4)]

    # inputs are of size 14 x 14
    layers += [ConvBlock(3, [256, 256, 1024], stage=4, block=1)]
    layers += [IdentityBlock(3, [256, 256, 1024], stage=4, block=2)]
    layers += [IdentityBlock(3, [256, 256, 1024], stage=4, block=3)]
    layers += [IdentityBlock(3, [256, 256, 1024], stage=4, block=4)]
    layers += [IdentityBlock(3, [256, 256, 1024], stage=4, block=5)]
    layers += [IdentityBlock(3, [256, 256, 1024], stage=4, block=6)]

    # inputs are of size 7 x 7
    layers += [ConvBlock(3, [512, 512, 2048], stage=5, block=1)]
    layers += [IdentityBlock(3, [512, 512, 2048], stage=5, block=2)]
    layers += [IdentityBlock(3, [512, 512, 2048], stage=5, block=3)]

    return tf.keras.Sequential(layers)


def resnet101_modified(input_tensor=None, input_shape=None, **kwargs):
    """
    The resnet101 from `tf.keras.applications.Resnet101` has a problem with the convolutional layers.
    It basically add bias terms to such layers followed by batch normalizations, which is not correct

    https://github.com/tensorflow/tensorflow/issues/37365

    This resnet 10 implementation provides a cleaner version

    """

    if input_tensor is None:
        input_tensor = tf.keras.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            input_tensor = tf.keras.Input(tensor=input_tensor, shape=input_shape)

    bn_axis = 3
    # inputs are of size 224 x 224 x 3
    layers = [input_tensor]
    layers += [
        Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            kernel_initializer="orthogonal",
            use_bias=False,
            trainable=True,
            kernel_regularizer=l2(weight_decay),
            padding="same",
            name="conv1/7x7_s2",
        )
    ]

    # inputs are of size 112 x 112 x 64
    layers += [BatchNormalization(axis=bn_axis, name="conv1/7x7_s2/bn")]
    layers += [Activation("relu")]
    layers += [MaxPooling2D((3, 3), strides=(2, 2))]

    # inputs are of size 56 x 56
    layers += [ConvBlock(3, [64, 64, 256], stage=2, block=1, strides=(1, 1))]
    layers += [IdentityBlock(3, [64, 64, 256], stage=2, block=2)]
    layers += [IdentityBlock(3, [64, 64, 256], stage=2, block=3)]

    # inputs are of size 28 x 28
    layers += [ConvBlock(3, [128, 128, 512], stage=3, block=1)]
    layers += [IdentityBlock(3, [128, 128, 512], stage=3, block=2)]
    layers += [IdentityBlock(3, [128, 128, 512], stage=3, block=3)]
    layers += [IdentityBlock(3, [128, 128, 512], stage=3, block=4)]

    # inputs are of size 14 x 14
    # 23 blocks here. That's the only difference from
    # resnet-101
    layers += [ConvBlock(3, [256, 256, 1024], stage=4, block=1)]
    for i in range(2, 24):
        layers += [IdentityBlock(3, [256, 256, 1024], stage=4, block=i)]

    # inputs are of size 7 x 7
    layers += [ConvBlock(3, [512, 512, 2048], stage=5, block=1)]
    layers += [IdentityBlock(3, [512, 512, 2048], stage=5, block=2)]
    layers += [IdentityBlock(3, [512, 512, 2048], stage=5, block=3)]

    return tf.keras.Sequential(layers)


if __name__ == "__main__":
    input_tensor = tf.keras.layers.InputLayer([112, 112, 3])
    model = resnet50_modified(input_tensor)

    print(len(model.variables))
    print(model.summary())
