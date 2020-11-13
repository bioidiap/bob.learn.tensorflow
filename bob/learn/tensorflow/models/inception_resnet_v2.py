# -*- coding: utf-8 -*-
"""Inception-ResNet-V2 MultiScale-Inception-ResNet-V2 models for Keras.
"""
import logging

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from ..utils import SequentialLayer

logger = logging.getLogger(__name__)


def Conv2D_BN(
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
    **kwargs,
):
    """Utility class to apply conv + BN.

    # Arguments
        x: input tensor.
        filters:
        kernel_size:
        strides:
        padding:
        activation:
        use_bias:

    Attributes
    ----------
    activation
        activation in `Conv2D`.
    filters
        filters in `Conv2D`.
    kernel_size
        kernel size as in `Conv2D`.
    padding
        padding mode in `Conv2D`.
    strides
        strides in `Conv2D`.
    use_bias
        whether to use a bias in `Conv2D`.
    name
        name of the ops; will become `name + '/Act'` for the activation
        and `name + '/BatchNorm'` for the batch norm layer.
    """

    layers = [
        Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name="Conv2D",
        )
    ]

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == "channels_first" else 3
        layers += [BatchNormalization(axis=bn_axis, scale=False, name="BatchNorm")]

    if activation is not None:
        layers += [Activation(activation, name="Act")]

    return SequentialLayer(layers, name=name, **kwargs)


class ScaledResidual(tf.keras.layers.Layer):
    """A scaled residual connection layer"""

    def __init__(self, scale, name="scaled_residual", **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = scale

    def call(self, inputs, training=None):
        return inputs[0] + inputs[1] * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale, "name": self.name})
        return config


class InceptionResnetBlock(tf.keras.layers.Layer):
    """An Inception-ResNet block.

    This class builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Attributes
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """

    def __init__(
        self,
        n_channels,
        scale,
        block_type,
        block_idx,
        activation="relu",
        n=1,
        name=None,
        **kwargs,
    ):
        name = name or block_type
        super().__init__(name=name, **kwargs)
        self.n_channels = n_channels
        self.scale = scale
        self.block_type = block_type
        self.block_idx = block_idx
        self.activation = activation
        self.n = n

        if block_type == "block35":
            branch_0 = [Conv2D_BN(32 // n, 1, name="Branch_0/Conv2d_1x1")]
            branch_1 = [Conv2D_BN(32 // n, 1, name="Branch_1/Conv2d_0a_1x1")]
            branch_1 += [Conv2D_BN(32 // n, 3, name="Branch_1/Conv2d_0b_3x3")]
            branch_2 = [Conv2D_BN(32 // n, 1, name="Branch_2/Conv2d_0a_1x1")]
            branch_2 += [Conv2D_BN(48 // n, 3, name="Branch_2/Conv2d_0b_3x3")]
            branch_2 += [Conv2D_BN(64 // n, 3, name="Branch_2/Conv2d_0c_3x3")]
            branches = [branch_0, branch_1, branch_2]
        elif block_type == "block17":
            branch_0 = [Conv2D_BN(192 // n, 1, name="Branch_0/Conv2d_1x1")]
            branch_1 = [Conv2D_BN(128 // n, 1, name="Branch_1/Conv2d_0a_1x1")]
            branch_1 += [Conv2D_BN(160 // n, (1, 7), name="Branch_1/Conv2d_0b_1x7")]
            branch_1 += [Conv2D_BN(192 // n, (7, 1), name="Branch_1/Conv2d_0c_7x1")]
            branches = [branch_0, branch_1]
        elif block_type == "block8":
            branch_0 = [Conv2D_BN(192 // n, 1, name="Branch_0/Conv2d_1x1")]
            branch_1 = [Conv2D_BN(192 // n, 1, name="Branch_1/Conv2d_0a_1x1")]
            branch_1 += [Conv2D_BN(224 // n, (1, 3), name="Branch_1/Conv2d_0b_1x3")]
            branch_1 += [Conv2D_BN(256 // n, (3, 1), name="Branch_1/Conv2d_0c_3x1")]
            branches = [branch_0, branch_1]
        else:
            raise ValueError(
                "Unknown Inception-ResNet block type. "
                'Expects "block35", "block17" or "block8", '
                "but got: " + str(block_type)
            )

        self.branches = branches

        channel_axis = 1 if K.image_data_format() == "channels_first" else 3
        self.concat = Concatenate(axis=channel_axis, name="concatenate")
        self.up_conv = Conv2D_BN(
            n_channels, 1, activation=None, use_bias=True, name="Conv2d_1x1"
        )

        self.residual = ScaledResidual(scale)
        self.act = lambda x: x
        if activation is not None:
            self.act = Activation(activation, name="act")

    def call(self, inputs, training=None):
        branch_outputs = []
        for branch in self.branches:
            x = inputs
            for layer in branch:
                x = layer(x, training=training)
            branch_outputs.append(x)

        mixed = self.concat(branch_outputs)
        up = self.up_conv(mixed, training=training)

        x = self.residual([inputs, up])
        x = self.act(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                name: getattr(self, name)
                for name in [
                    "n_channels",
                    "scale",
                    "block_type",
                    "block_idx",
                    "activation",
                    "n",
                    "name",
                ]
            }
        )
        return config


class ReductionA(tf.keras.layers.Layer):
    """A Reduction A block for InceptionResnetV2"""

    def __init__(
        self,
        padding,
        k=256,
        kl=256,
        km=384,
        n=384,
        use_atrous=False,
        name="reduction_a",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.padding = padding
        self.k = k
        self.kl = kl
        self.km = km
        self.n = n
        self.use_atrous = use_atrous

        branch_1 = [
            Conv2D_BN(
                n,
                3,
                strides=1 if use_atrous else 2,
                padding=padding,
                name="Branch_0/Conv2d_1a_3x3",
            )
        ]

        branch_2 = [
            Conv2D_BN(k, 1, name="Branch_1/Conv2d_0a_1x1"),
            Conv2D_BN(kl, 3, name="Branch_1/Conv2d_0b_3x3"),
            Conv2D_BN(
                km,
                3,
                strides=1 if use_atrous else 2,
                padding=padding,
                name="Branch_1/Conv2d_1a_3x3",
            ),
        ]

        branch_pool = [
            MaxPool2D(
                3,
                strides=1 if use_atrous else 2,
                padding=padding,
                name="Branch_2/MaxPool_1a_3x3",
            )
        ]
        self.branches = [branch_1, branch_2, branch_pool]
        channel_axis = 1 if K.image_data_format() == "channels_first" else 3
        self.concat = Concatenate(axis=channel_axis, name=f"{name}/mixed")

    def call(self, inputs, training=None):
        branch_outputs = []
        for branch in self.branches:
            x = inputs
            for layer in branch:
                try:
                    x = layer(x, training=training)
                except TypeError:
                    x = layer(x)
            branch_outputs.append(x)

        return self.concat(branch_outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                name: getattr(self, name)
                for name in ["padding", "k", "kl", "km", "n", "use_atrous", "name"]
            }
        )
        return config


class ReductionB(tf.keras.layers.Layer):
    """A Reduction B block for InceptionResnetV2"""

    def __init__(
        self,
        padding,
        k=256,
        kl=288,
        km=320,
        n=256,
        no=384,
        p=256,
        pq=288,
        name="reduction_b",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.padding = padding
        self.k = k
        self.kl = kl
        self.km = km
        self.n = n
        self.no = no
        self.p = p
        self.pq = pq

        branch_1 = [
            Conv2D_BN(n, 1, name="Branch_0/Conv2d_0a_1x1"),
            Conv2D_BN(no, 3, strides=2, padding=padding, name="Branch_0/Conv2d_1a_3x3"),
        ]

        branch_2 = [
            Conv2D_BN(p, 1, name="Branch_1/Conv2d_0a_1x1"),
            Conv2D_BN(pq, 3, strides=2, padding=padding, name="Branch_1/Conv2d_1a_3x3"),
        ]

        branch_3 = [
            Conv2D_BN(k, 1, name="Branch_2/Conv2d_0a_1x1"),
            Conv2D_BN(kl, 3, name="Branch_2/Conv2d_0b_3x3"),
            Conv2D_BN(km, 3, strides=2, padding=padding, name="Branch_2/Conv2d_1a_3x3"),
        ]

        branch_pool = [
            MaxPool2D(3, strides=2, padding=padding, name="Branch_3/MaxPool_1a_3x3")
        ]
        self.branches = [branch_1, branch_2, branch_3, branch_pool]
        channel_axis = 1 if K.image_data_format() == "channels_first" else 3
        self.concat = Concatenate(axis=channel_axis, name=f"{name}/mixed")

    def call(self, inputs, training=None):
        branch_outputs = []
        for branch in self.branches:
            x = inputs
            for layer in branch:
                try:
                    x = layer(x, training=training)
                except TypeError:
                    x = layer(x)
            branch_outputs.append(x)

        return self.concat(branch_outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                name: getattr(self, name)
                for name in ["padding", "k", "kl", "km", "n", "no", "p", "pq", "name"]
            }
        )
        return config


class InceptionA(tf.keras.layers.Layer):
    def __init__(self, pool_filters, name="inception_a", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pool_filters = pool_filters

        self.branch1x1 = Conv2D_BN(
            96, kernel_size=1, padding="same", name="Branch_0/Conv2d_1x1"
        )

        self.branch3x3dbl_1 = Conv2D_BN(
            64, kernel_size=1, padding="same", name="Branch_2/Conv2d_0a_1x1"
        )
        self.branch3x3dbl_2 = Conv2D_BN(
            96, kernel_size=3, padding="same", name="Branch_2/Conv2d_0b_3x3"
        )
        self.branch3x3dbl_3 = Conv2D_BN(
            96, kernel_size=3, padding="same", name="Branch_2/Conv2d_0c_3x3"
        )

        self.branch5x5_1 = Conv2D_BN(
            48, kernel_size=1, padding="same", name="Branch_1/Conv2d_0a_1x1"
        )
        self.branch5x5_2 = Conv2D_BN(
            64, kernel_size=5, padding="same", name="Branch_1/Conv2d_0b_5x5"
        )

        self.branch_pool_1 = AvgPool2D(
            pool_size=3, strides=1, padding="same", name="Branch_3/AvgPool_0a_3x3"
        )
        self.branch_pool_2 = Conv2D_BN(
            pool_filters, kernel_size=1, padding="same", name="Branch_3/Conv2d_0b_1x1"
        )

        channel_axis = 1 if K.image_data_format() == "channels_first" else 3
        self.concat = Concatenate(axis=channel_axis)

    def call(self, inputs, training=None):
        branch1x1 = self.branch1x1(inputs)

        branch3x3dbl = self.branch3x3dbl_1(inputs)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch5x5 = self.branch5x5_1(inputs)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.branch_pool_1(inputs)
        branch_pool = self.branch_pool_2(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return self.concat(outputs)

    def get_config(self):
        config = super().get_config()
        config.update({"pool_filters": self.pool_filters, "name": self.name})
        return config


def InceptionResNetV2(
    include_top=True,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    bottleneck=False,
    dropout_rate=0.2,
    name="InceptionResnetV2",
    **kwargs,
):
    """Instantiates the Inception-ResNet v2 architecture.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor (i.e. output of `tf.keras.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if input_tensor is None:
        img_input = tf.keras.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = tf.keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    layers = [
        # Stem block: 35 x 35 x 192
        Conv2D_BN(32, 3, strides=2, padding="valid", name="Conv2d_1a_3x3"),
        Conv2D_BN(32, 3, padding="valid", name="Conv2d_2a_3x3"),
        Conv2D_BN(64, 3, name="Conv2d_2b_3x3"),
        MaxPool2D(3, strides=2, name="MaxPool_3a_3x3"),
        Conv2D_BN(80, 1, padding="valid", name="Conv2d_3b_1x1"),
        Conv2D_BN(192, 3, padding="valid", name="Conv2d_4a_3x3"),
        MaxPool2D(3, strides=2, name="MaxPool_5a_3x3"),
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        InceptionA(pool_filters=64, name="Mixed_5b"),
    ]

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        layers.append(
            InceptionResnetBlock(
                n_channels=320,
                scale=0.17,
                block_type="block35",
                block_idx=block_idx,
                name=f"block35_{block_idx}",
            )
        )

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    layers.append(
        ReductionA(
            padding="valid",
            n=384,
            k=256,
            kl=256,
            km=384,
            use_atrous=False,
            name="Mixed_6a",
        )
    )

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        layers.append(
            InceptionResnetBlock(
                n_channels=1088,
                scale=0.1,
                block_type="block17",
                block_idx=block_idx,
                name=f"block17_{block_idx}",
            )
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    layers.append(
        ReductionB(
            padding="valid",
            n=256,
            no=384,
            p=256,
            pq=288,
            k=256,
            kl=288,
            km=320,
            name="Mixed_7a",
        )
    )

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        layers.append(
            InceptionResnetBlock(
                n_channels=2080,
                scale=0.2,
                block_type="block8",
                block_idx=block_idx,
                name=f"block8_{block_idx}",
            )
        )
    layers.append(
        InceptionResnetBlock(
            n_channels=2080,
            scale=1.0,
            activation=None,
            block_type="block8",
            block_idx=10,
            name=f"block8_{block_idx+1}",
        )
    )

    # Final convolution block: 8 x 8 x 1536
    layers.append(Conv2D_BN(1536, 1, name="Conv2d_7b_1x1"))

    if (include_top and pooling is None) or (bottleneck):
        pooling = "avg"

    if pooling == "avg":
        layers.append(GlobalAvgPool2D())
    elif pooling == "max":
        layers.append(GlobalMaxPool2D())

    if bottleneck:
        layers.append(Dropout(dropout_rate, name="Dropout"))
        layers.append(Dense(128, use_bias=False, name="Bottleneck"))
        layers.append(
            BatchNormalization(axis=-1, scale=False, name="Bottleneck/BatchNorm")
        )

    # Classification block
    if include_top:
        layers.append(Dense(classes, name="logits"))

    # Create model and call it on input to create its variables.
    model = Sequential(layers, name=name, **kwargs)
    model(img_input)

    return model


def MultiScaleInceptionResNetV2(
    scale=0.17,
    repeat=3,
    classes=1,
    dropout_rate=0.2,
    input_tensor=None,
    input_shape=None,
    align_feature_maps=False,
    name="InceptionResnetV2",
    **kwargs,
):
    """A multi-scale architecture inspired from InceptionResNetV2"""
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    padding = "SAME" if align_feature_maps else "VALID"
    name = name or "InceptionResnetV2"

    with tf.compat.v1.name_scope(name, "InceptionResnetV2", [img_input]):
        # convert colors from RGB to a learned color space and batch norm inputs
        # 224, 224, 4
        net = Conv2D_BN(
            4, 1, strides=1, padding="same", activation=None, name="Conv2d_1i_1x1"
        )(img_input)

        # reduction_a: 111, 111, 32
        net = ReductionA(
            padding=padding, k=8, kl=12, km=14, n=14, name="Reduction_a_1"
        )(net)

        # 111, 111, 32
        for block_idx in range(1, 1 + repeat):
            net = InceptionResnetBlock(
                n_channels=32,
                scale=scale,
                block_type="block35",
                block_idx=block_idx,
                activation="relu",
                n=2,
                name=f"Repeat/block35_{block_idx}",
            )(net)
        scale_1 = net

        # 55, 55, 96
        net = ReductionA(
            padding=padding, k=32, kl=32, km=32, n=32, name="Reduction_a_2"
        )(net)

        # 55, 55, 96
        for block_idx in range(1, 1 + repeat):
            net = InceptionResnetBlock(
                n_channels=96,
                scale=scale,
                block_type="block17",
                block_idx=block_idx,
                n=2,
                activation="relu",
                name=f"Repeat_1/block17_{block_idx}",
            )(net)
        scale_2 = net

        # 27, 27, 344
        net = ReductionB(
            padding, k=64, kl=72, km=80, n=64, no=96, p=64, pq=72, name="Reduction_b"
        )(net)

        # 27, 27, 344
        for block_idx in range(1, 1 + repeat):
            net = InceptionResnetBlock(
                n_channels=344,
                scale=scale,
                block_type="block8",
                block_idx=block_idx,
                n=1,
                activation="relu",
                name=f"Repeat_2/block8_{block_idx}",
            )(net)
        scale_3 = net

        # 27, 27, 32
        scale_1 = AvgPool2D(3, strides=2, padding=padding, name="Merge/AvgPool_1a")(
            scale_1
        )
        scale_1 = AvgPool2D(3, strides=2, padding=padding, name="Merge/AvgPool_1b")(
            scale_1
        )
        # 27, 27, 96
        scale_2 = AvgPool2D(3, strides=2, padding=padding, name="Merge/AvgPool_2")(
            scale_2
        )
        # 27, 27, 344
        scale_3 = scale_3

        # 27, 27, 472 * 3
        channel_axis = 1 if K.image_data_format() == "channels_first" else 3
        net = Concatenate(axis=channel_axis, name="Merge/concat")(
            [scale_1, scale_2, scale_3]
        )

        # 27, 27, 256
        net = Conv2D_BN(256, 1, name="Merge/Conv2d_1")(net)

        # 13, 13, 256
        net = AvgPool2D(3, strides=2, padding=padding, name="Merge/AvgPool_3")(net)

        # 13, 13, 128
        net = Conv2D_BN(128, 1, name="Merge/Conv2d_2")(net)

        net = Dropout(dropout_rate, name="Merge/Dropout")(net)

        # 13, 13, classes
        net = Conv2D(classes, 1, padding="same", name="Pixel_Logits")(net)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, net, name=name, **kwargs)

    return model


if __name__ == "__main__":
    import pkg_resources  # noqa: F401
    from tabulate import tabulate

    from bob.learn.tensorflow.utils import model_summary

    def print_model(inputs, outputs, name=None):
        print("")
        print("===============")
        print(name)
        print("===============")
        model = tf.keras.Model(inputs, outputs)
        rows = model_summary(model, do_print=True)
        del rows[-2]
        print(tabulate(rows, headers="firstrow", tablefmt="latex"))

    # model = InceptionResNetV2(input_shape=(299, 299, 3))
    # inputs = tf.keras.Input((299, 299, 3))
    # outputs = model.call(inputs)
    # print_model(inputs, outputs)

    # inputs = tf.keras.Input((299, 299, 3))
    # outputs = model.get_layer("conv2d_bn").call(inputs)
    # print_model(inputs, outputs, name="conv2d_bn")

    # inputs = tf.keras.Input((35, 35, 192))
    # outputs = model.get_layer("inception_a").call(inputs)
    # print_model(inputs, outputs, name="inception_a")

    # inputs = tf.keras.Input((35, 35, 320))
    # outputs = model.get_layer("block35_1").call(inputs)
    # print_model(inputs, outputs, name="block35_1")

    # inputs = tf.keras.Input((17, 17, 1088))
    # outputs = model.get_layer("block17_1").call(inputs)
    # print_model(inputs, outputs, name="block17_1")

    # inputs = tf.keras.Input((8, 8, 2080))
    # outputs = model.get_layer("block8_1").call(inputs)
    # print_model(inputs, outputs, name="block8_1")

    # inputs = tf.keras.Input((35, 35, 320))
    # outputs = model.get_layer("reduction_a").call(inputs)
    # print_model(inputs, outputs, name="reduction_a")

    # inputs = tf.keras.Input((17, 17, 1088))
    # outputs = model.get_layer("reduction_b").call(inputs)
    # print_model(inputs, outputs, name="reduction_b")

    model = MultiScaleInceptionResNetV2(input_shape=(224, 224, 3))
    inputs = tf.keras.Input((224, 224, 3))
    outputs = model.call(inputs)
    print_model(inputs, outputs)

    # inputs = tf.keras.Input((224, 224, 3))
    # outputs = model.get_layer("Conv2d_1i_1x1").call(inputs)
    # print_model(inputs, outputs)
