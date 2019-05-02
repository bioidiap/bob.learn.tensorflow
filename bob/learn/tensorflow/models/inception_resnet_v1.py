# -*- coding: utf-8 -*-
"""Inception-ResNet V1 model for Keras.
# Reference
http://arxiv.org/abs/1602.07261
https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
https://github.com/myutwo150/keras-inception-resnet-v2/blob/master/inception_resnet_v2.py
"""
from functools import partial

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def scaling(x, scale):
    return x * scale


def conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
    timedistributed=False,
    kernel_regularizer=None,
    training=False,
):
    if not timedistributed:

        def MyTimeDistributed(x):
            return x

    else:
        MyTimeDistributed = TimeDistributed

    x = MyTimeDistributed(
        Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            name=name,
        )
    )(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == "channels_first" else 3
        bn_name = _generate_layer_name("BatchNorm", prefix=name)
        x = BatchNormalization(
            axis=bn_axis + 1 if timedistributed else bn_axis,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=bn_name,
        )(x, training=training)
    if activation is not None:
        ac_name = _generate_layer_name("Activation", prefix=name)
        x = MyTimeDistributed(Activation(activation, name=ac_name))(x)
    return x


def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return "_".join((prefix, name))
    return "_".join((prefix, "Branch", str(branch_idx), name))


def _inception_resnet_block(
    x,
    scale,
    block_type,
    block_idx,
    activation="relu",
    timedistributed=False,
    kernel_regularizer=None,
    training=False,
):
    if not timedistributed:

        def MyTimeDistributed(x):
            return x

    else:
        MyTimeDistributed = TimeDistributed

    channel_axis = 1 if K.image_data_format() == "channels_first" else 3
    if block_idx is None:
        prefix = None
    else:
        prefix = "_".join((block_type, str(block_idx)))
    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == "Block35":
        branch_0 = conv2d_bn(
            x,
            32,
            1,
            name=name_fmt("Conv2d_1x1", 0),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            x,
            32,
            1,
            name=name_fmt("Conv2d_0a_1x1", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            branch_1,
            32,
            3,
            name=name_fmt("Conv2d_0b_3x3", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_2 = conv2d_bn(
            x,
            32,
            1,
            name=name_fmt("Conv2d_0a_1x1", 2),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_2 = conv2d_bn(
            branch_2,
            32,
            3,
            name=name_fmt("Conv2d_0b_3x3", 2),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_2 = conv2d_bn(
            branch_2,
            32,
            3,
            name=name_fmt("Conv2d_0c_3x3", 2),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "Block17":
        branch_0 = conv2d_bn(
            x,
            128,
            1,
            name=name_fmt("Conv2d_1x1", 0),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            x,
            128,
            1,
            name=name_fmt("Conv2d_0a_1x1", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            branch_1,
            128,
            [1, 7],
            name=name_fmt("Conv2d_0b_1x7", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            branch_1,
            128,
            [7, 1],
            name=name_fmt("Conv2d_0c_7x1", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branches = [branch_0, branch_1]
    elif block_type == "Block8":
        branch_0 = conv2d_bn(
            x,
            192,
            1,
            name=name_fmt("Conv2d_1x1", 0),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            x,
            192,
            1,
            name=name_fmt("Conv2d_0a_1x1", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            branch_1,
            192,
            [1, 3],
            name=name_fmt("Conv2d_0b_1x3", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branch_1 = conv2d_bn(
            branch_1,
            192,
            [3, 1],
            name=name_fmt("Conv2d_0c_3x1", 1),
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
        branches = [branch_0, branch_1]
    else:
        raise ValueError(
            "Unknown Inception-ResNet block type. "
            'Expects "Block35", "Block17" or "Block8", '
            "but got: " + str(block_type)
        )

    if timedistributed:
        channel_axis += 1

    mixed = Concatenate(axis=channel_axis, name=name_fmt("Concatenate"))(branches)
    up = conv2d_bn(
        mixed,
        K.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=name_fmt("Conv2d_1x1"),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    up = MyTimeDistributed(
        Lambda(
            scaling,
            output_shape=K.int_shape(up)[2 if timedistributed else 1 :],
            arguments={"scale": scale},
        )
    )(up)
    x = Add()([x, up])
    if activation is not None:
        x = MyTimeDistributed(Activation(activation, name=name_fmt("Activation")))(x)
    return x


def InceptionResNetV1(
    input_shape=(160, 160, 3),
    inputs=None,
    classes=128,
    dropout_keep_prob=0.8,
    weight_decay=1e-5,
    weights_path=None,
    timedistributed=False,
    training=False,
):
    if not timedistributed:

        def MyTimeDistributed(x):
            return x

    else:
        MyTimeDistributed = TimeDistributed
        input_shape = [None] + list(input_shape)

    if weight_decay is None:
        kernel_regularizer = None
    else:
        kernel_regularizer = tf.keras.regularizers.l2(l=weight_decay)

    if inputs is None:
        inputs = Input(shape=input_shape)
    x = conv2d_bn(
        inputs,
        32,
        3,
        strides=2,
        padding="valid",
        name="Conv2d_1a_3x3",
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    x = conv2d_bn(
        x,
        32,
        3,
        padding="valid",
        name="Conv2d_2a_3x3",
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    x = conv2d_bn(
        x,
        64,
        3,
        name="Conv2d_2b_3x3",
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    x = MyTimeDistributed(MaxPooling2D(3, strides=2, name="MaxPool_3a_3x3"))(x)
    x = conv2d_bn(
        x,
        80,
        1,
        padding="valid",
        name="Conv2d_3b_1x1",
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    x = conv2d_bn(
        x,
        192,
        3,
        padding="valid",
        name="Conv2d_4a_3x3",
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    x = conv2d_bn(
        x,
        256,
        3,
        strides=2,
        padding="valid",
        name="Conv2d_4b_3x3",
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )

    # 5x Block35 (Inception-ResNet-A block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(
            x,
            scale=0.17,
            block_type="Block35",
            block_idx=block_idx,
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )

    # Mixed 6a (Reduction-A block):
    channel_axis = 1 if K.image_data_format() == "channels_first" else 3
    name_fmt = partial(_generate_layer_name, prefix="Mixed_6a")
    branch_0 = conv2d_bn(
        x,
        384,
        3,
        strides=2,
        padding="valid",
        name=name_fmt("Conv2d_1a_3x3", 0),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_1 = conv2d_bn(
        x,
        192,
        1,
        name=name_fmt("Conv2d_0a_1x1", 1),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_1 = conv2d_bn(
        branch_1,
        192,
        3,
        name=name_fmt("Conv2d_0b_3x3", 1),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_1 = conv2d_bn(
        branch_1,
        256,
        3,
        strides=2,
        padding="valid",
        name=name_fmt("Conv2d_1a_3x3", 1),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_pool = MyTimeDistributed(
        MaxPooling2D(3, strides=2, padding="valid", name=name_fmt("MaxPool_1a_3x3", 2))
    )(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(
        axis=channel_axis + 1 if timedistributed else channel_axis, name="Mixed_6a"
    )(branches)

    # 10x Block17 (Inception-ResNet-B block):
    for block_idx in range(1, 11):
        x = _inception_resnet_block(
            x,
            scale=0.1,
            block_type="Block17",
            block_idx=block_idx,
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = partial(_generate_layer_name, prefix="Mixed_7a")
    branch_0 = conv2d_bn(
        x,
        256,
        1,
        name=name_fmt("Conv2d_0a_1x1", 0),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_0 = conv2d_bn(
        branch_0,
        384,
        3,
        strides=2,
        padding="valid",
        name=name_fmt("Conv2d_1a_3x3", 0),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_1 = conv2d_bn(
        x,
        256,
        1,
        name=name_fmt("Conv2d_0a_1x1", 1),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_1 = conv2d_bn(
        branch_1,
        256,
        3,
        strides=2,
        padding="valid",
        name=name_fmt("Conv2d_1a_3x3", 1),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_2 = conv2d_bn(
        x,
        256,
        1,
        name=name_fmt("Conv2d_0a_1x1", 2),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_2 = conv2d_bn(
        branch_2,
        256,
        3,
        name=name_fmt("Conv2d_0b_3x3", 2),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_2 = conv2d_bn(
        branch_2,
        256,
        3,
        strides=2,
        padding="valid",
        name=name_fmt("Conv2d_1a_3x3", 2),
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )
    branch_pool = MyTimeDistributed(
        MaxPooling2D(3, strides=2, padding="valid", name=name_fmt("MaxPool_1a_3x3", 3))
    )(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(
        axis=channel_axis + 1 if timedistributed else channel_axis, name="Mixed_7a"
    )(branches)

    # 5x Block8 (Inception-ResNet-C block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(
            x,
            scale=0.2,
            block_type="Block8",
            block_idx=block_idx,
            kernel_regularizer=kernel_regularizer,
            timedistributed=timedistributed,
            training=training,
        )
    x = _inception_resnet_block(
        x,
        scale=1.0,
        activation=None,
        block_type="Block8",
        block_idx=6,
        kernel_regularizer=kernel_regularizer,
        timedistributed=timedistributed,
        training=training,
    )

    # Classification block
    x = MyTimeDistributed(GlobalAveragePooling2D(name="AvgPool"))(x)
    x = MyTimeDistributed(Dropout(1.0 - dropout_keep_prob, name="Dropout"))(
        x, training=training
    )
    # Bottleneck
    x = MyTimeDistributed(
        Dense(
            classes,
            use_bias=False,
            name="Bottleneck",
            kernel_regularizer=kernel_regularizer,
        )
    )(x)
    bn_name = _generate_layer_name("BatchNorm", prefix="Bottleneck")
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name=bn_name)(
        x, training=training
    )

    if timedistributed:
        return x
    # Create model
    model = Model(inputs, x, name="inception_resnet_v1")
    if weights_path is not None:
        logger.info("restoring model weights from %s", weights_path)
        model.load_weights(weights_path)

    return model


if __name__ == "__main__":
    import pkg_resources

    tf.enable_eager_execution()
    import numpy as np
    from bob.extension import rc

    def input_fn():
        features = {
            "data": np.empty((100, 160, 160, 3), dtype="float32"),
            "key": "path",
        }
        labels = {"bio": 10, "pad": int(True)}
        dataset = tf.data.Dataset.from_tensors((features, labels))
        return dataset.repeat(2).batch(1)

    input_shape = (None, 160, 160, 3)
    inputs = tf.keras.layers.Input(input_shape, name="data")
    key = tf.keras.layers.Input((None,), name="key")
    embedding = InceptionResNetV1(timedistributed=True, inputs=inputs)
    tf.keras.Model(inputs, embedding, name="inception_resnet_v1").load_weights(
        rc["bob.learn.tensorflow.facenet_keras_weights"]
    )
    model = tf.keras.layers.LSTM(128)(embedding)
    bio = tf.keras.layers.Dense(10, activation="softmax", name="bio")(model)
    pad = tf.keras.layers.Dense(2, activation="softmax", name="pad")(model)
    model = tf.keras.Model(inputs=[inputs, key], outputs=[bio, pad])
    # model.build(input_shape=[None] + list(input_shape))
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss="categorical_crossentropy",
        loss_weights=[0.5, 0.5],
        metrics=["accuracy"],
    )

    # model.fit(input_fn(), steps_per_epoch=1)

    estimator = tf.keras.estimator.model_to_estimator(
        model,
        model_dir="/scratch/amohammadi/tmp/keras_model",
        # config=run_config,
    )

    estimator.train(input_fn)
