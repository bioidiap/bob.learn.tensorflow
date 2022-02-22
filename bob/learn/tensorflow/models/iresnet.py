"""iResNet models for Keras.
Adapted from insightface/recognition/arcface_torch/backbones/iresnet.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = ["iresnet18", "iresnet34", "iresnet50", "iresnet100", "iresnet200"]


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.L2(l2_weight_decay) if use_l2_regularizer else None


def _gen_initializer():
    return tf.keras.initializers.RandomNormal(mean=0, stddev=0.1)


def conv3x3(filters, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same" if dilation else "valid",
        groups=groups,
        use_bias=False,
        dilation_rate=dilation,
        kernel_initializer=_gen_initializer(),
    )


def conv1x1(filters, stride=1):
    """1x1 convolution"""
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=stride,
        padding="valid",
        use_bias=False,
        kernel_initializer=_gen_initializer(),
    )


def IBasicBlock(
    x, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1
):
    if groups != 1 or base_width != 64:
        raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1:
        raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    bn1 = tf.keras.layers.BatchNormalization(
        scale=False,
        momentum=0.9,
        epsilon=1e-05,
    )
    conv1 = conv3x3(planes)
    bn2 = tf.keras.layers.BatchNormalization(
        scale=False,
        momentum=0.9,
        epsilon=1e-05,
    )
    prelu = tf.keras.layers.PReLU(
        alpha_initializer=tf.keras.initializers.Constant(0.25), shared_axes=[1, 2]
    )
    conv2 = conv3x3(planes, stride=stride)
    bn3 = tf.keras.layers.BatchNormalization(
        scale=False,
        momentum=0.9,
        epsilon=1e-05,
    )

    identity = x
    out = bn1(x)
    out = conv1(out)
    out = bn2(out)
    out = prelu(out)
    out = conv2(out)
    out = bn3(out)
    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)
    out += identity
    return out


def _make_layer(
    x, dilation, groups, base_width, block, planes, blocks, stride=1, dilate=False
):
    downsample = None
    previous_dilation = dilation
    if dilate:
        dilation *= stride
        stride = 1
    if stride != 1 or x.shape[-1] != planes:
        downsample = [
            conv1x1(planes, stride),
            tf.keras.layers.BatchNormalization(
                scale=False,
                momentum=0.9,
                epsilon=1e-05,
            ),
        ]
    x = block(x, planes, stride, downsample, groups, base_width, previous_dilation)
    for _ in range(1, blocks):
        x = block(x, planes, groups=groups, base_width=base_width, dilation=dilation)

    return x, dilation


def IResNet(
    name,
    input_shape,
    block,
    layers,
    groups=1,
    width_per_group=64,
    replace_stride_with_dilation=None,
):
    x = img_input = tf.keras.layers.Input(shape=input_shape)
    inplanes = 64
    dilation = 1
    if replace_stride_with_dilation is None:
        replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
        raise ValueError(
            "replace_stride_with_dilation should be None "
            "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
        )
    groups = groups
    base_width = width_per_group
    conv1 = conv3x3(inplanes, stride=1, dilation=1)
    bn1 = tf.keras.layers.BatchNormalization(
        scale=False,
        momentum=0.9,
        epsilon=1e-05,
    )
    prelu = tf.keras.layers.PReLU(
        alpha_initializer=tf.keras.initializers.Constant(0.25), shared_axes=[1, 2]
    )

    x = conv1(x)
    x = bn1(x)
    x = prelu(x)

    x, dilation = _make_layer(
        x=x,
        dilation=dilation,
        base_width=base_width,
        groups=groups,
        block=block,
        planes=64,
        blocks=layers[0],
        stride=2,
    )
    x, dilation = _make_layer(
        x=x,
        dilation=dilation,
        base_width=base_width,
        groups=groups,
        block=block,
        planes=128,
        blocks=layers[1],
        stride=2,
        dilate=replace_stride_with_dilation[0],
    )
    x, dilation = _make_layer(
        x=x,
        dilation=dilation,
        base_width=base_width,
        groups=groups,
        block=block,
        planes=256,
        blocks=layers[2],
        stride=2,
        dilate=replace_stride_with_dilation[1],
    )
    x, dilation = _make_layer(
        x=x,
        dilation=dilation,
        base_width=base_width,
        groups=groups,
        block=block,
        planes=512,
        blocks=layers[3],
        stride=2,
        dilate=replace_stride_with_dilation[2],
    )

    return tf.keras.Model(img_input, x, name=name)


def iresnet18(input_shape, **kwargs):
    return IResNet(
        name="iresnet18",
        input_shape=input_shape,
        block=IBasicBlock,
        layers=[2, 2, 2, 2],
        **kwargs,
    )


def iresnet34(input_shape, **kwargs):
    return IResNet(
        name="iresnet34",
        input_shape=input_shape,
        block=IBasicBlock,
        layers=[3, 4, 6, 3],
        **kwargs,
    )


def iresnet50(input_shape, **kwargs):
    return IResNet(
        name="iresnet50",
        input_shape=input_shape,
        block=IBasicBlock,
        layers=[3, 4, 14, 3],
        **kwargs,
    )


def iresnet100(input_shape, **kwargs):
    return IResNet(
        name="iresnet100",
        input_shape=input_shape,
        block=IBasicBlock,
        layers=[3, 13, 30, 3],
        **kwargs,
    )


def iresnet200(input_shape, **kwargs):
    return IResNet(
        name="iresnet200",
        input_shape=input_shape,
        block=IBasicBlock,
        layers=[6, 26, 60, 6],
        **kwargs,
    )


if __name__ == "__main__":
    model = iresnet50((112, 112, 3))
    model.summary()
    tf.keras.utils.plot_model(
        model, "keras_model.svg", show_shapes=True, expand_nested=True, dpi=300
    )
