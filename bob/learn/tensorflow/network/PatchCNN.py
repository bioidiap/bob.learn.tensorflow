"""Path-based CNN Estimator for "Face Anti-spoofing Using Patch and Depth-based
CNNs".

The architecture is:

+--------------+---------------+---------------+
| Layer        | Filter/Stride | Output Size   |
+--------------+---------------+---------------+
| Conv-1       | 5 x 5/1       | 96 x 96 x 50  |
| BN-1         |               | 96 x 96 x 50  |
| MaxPooling-1 | 2 x 2/2       | 48 x 48 x 50  |
+--------------+---------------+---------------+
| Conv-2       | 3 x 3/1       | 48 x 48 x 100 |
| BN-2         |               | 48 x 48 x 100 |
| MaxPooling-2 | 2 x 2/2       | 24 x 24 x 100 |
+--------------+---------------+---------------+
| Conv-3       | 3 x 3/1       | 24 x 24 x 150 |
| BN-3         |               | 24 x 24 x 150 |
| MaxPooling-3 | 3 x 3/2       | 12 x 12 x 150 |
+--------------+---------------+---------------+
| Conv-4       | 3 x 3/1       | 12 x 12 x 200 |
| BN-4         |               | 12 x 12 x 200 |
| MaxPooling-4 | 2 x 2/2       | 6 x 6 x 200   |
+--------------+---------------+---------------+
| Conv-5       | 3 x 3/1       | 6 x 6 x 250   |
| BN-5         |               | 6 x 6 x 250   |
| MaxPooling-5 | 2 x 2/2       | 3 x 3 x 250   |
+--------------+---------------+---------------+
| FC-1         | 3 x 3/1       | 1 x 1 x 1000  |
| BN-6         |               | 1 x 1 x 1000  |
| Dropout      | 0.5           | 1 x 1 x 1000  |
+--------------+---------------+---------------+
| FC-2         | 1 x 1/1       | 1 x 1 x 400   |
| BN-7         |               | 1 x 1 x 400   |
| FC-3         | 1 x 1/1       | 1 x 1 x 2     |
+--------------+---------------+---------------+

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def base_architecture(input_layer, mode, data_format, **kwargs):
    # Keep track of all the endpoints
    endpoints = {}
    bn_axis = 1 if data_format.lower() == 'channels_first' else -1
    training = mode == tf.estimator.ModeKeys.TRAIN

    # ======================
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=50,
        kernel_size=(5, 5),
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)
    endpoints['Conv-1'] = conv1

    # Batch Normalization #1
    bn1 = tf.layers.batch_normalization(conv1, axis=bn_axis, training=training)
    endpoints['BN-1'] = bn1

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=bn1, pool_size=[2, 2], strides=2, data_format=data_format)
    endpoints['MaxPooling-1'] = pool1

    # ======================
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=100,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)
    endpoints['Conv-2'] = conv2

    # Batch Normalization #2
    bn2 = tf.layers.batch_normalization(conv2, axis=bn_axis, training=training)
    endpoints['BN-2'] = bn2

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=bn2, pool_size=[2, 2], strides=2, data_format=data_format)
    endpoints['MaxPooling-2'] = pool2

    # ======================
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=150,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)
    endpoints['Conv-3'] = conv3

    # Batch Normalization #3
    bn3 = tf.layers.batch_normalization(conv3, axis=bn_axis, training=training)
    endpoints['BN-3'] = bn3

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(
        inputs=bn3, pool_size=[3, 3], strides=2, data_format=data_format)
    endpoints['MaxPooling-3'] = pool3

    # ======================
    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=200,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)
    endpoints['Conv-4'] = conv4

    # Batch Normalization #4
    bn4 = tf.layers.batch_normalization(conv4, axis=bn_axis, training=training)
    endpoints['BN-4'] = bn4

    # Pooling Layer #4
    pool4 = tf.layers.max_pooling2d(
        inputs=bn4, pool_size=[2, 2], strides=2, data_format=data_format)
    endpoints['MaxPooling-4'] = pool4

    # ======================
    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=pool4,
        filters=250,
        kernel_size=(3, 3),
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)
    endpoints['Conv-5'] = conv5

    # Batch Normalization #5
    bn5 = tf.layers.batch_normalization(conv5, axis=bn_axis, training=training)
    endpoints['BN-5'] = bn5

    # Pooling Layer #5
    pool5 = tf.layers.max_pooling2d(
        inputs=bn5, pool_size=[2, 2], strides=2, data_format=data_format)
    endpoints['MaxPooling-5'] = pool5

    # Flatten tensor into a batch of vectors
    pool5_flat = tf.layers.flatten(pool5)
    endpoints['MaxPooling-5-Flat'] = pool5_flat

    # ========================
    # Fully Connected Layer #1
    fc_1 = tf.layers.dense(
        inputs=pool5_flat, units=1000, activation=tf.nn.relu)
    endpoints['FC-1'] = fc_1

    # Batch Normalization #6
    bn6 = tf.layers.batch_normalization(fc_1, axis=bn_axis, training=training)
    endpoints['BN-6'] = bn6

    # Dropout
    dropout = tf.layers.dropout(inputs=bn6, rate=0.5, training=training)
    endpoints['dropout'] = dropout

    # ========================
    # Fully Connected Layer #2
    fc_2 = tf.layers.dense(inputs=dropout, units=400, activation=tf.nn.relu)
    endpoints['FC-2'] = fc_2

    # Batch Normalization #7
    bn7 = tf.layers.batch_normalization(fc_2, axis=bn_axis, training=training)
    endpoints['BN-7'] = bn7

    return bn7, endpoints


def architecture(input_layer,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 n_classes=2,
                 data_format='channels_last',
                 reuse=False,
                 **kwargs):

    with tf.variable_scope('PatchCNN', reuse=reuse):

        bn7, endpoints = base_architecture(input_layer, mode, data_format)
        # Logits layer
        logits = tf.layers.dense(inputs=bn7, units=n_classes)
        endpoints['FC-3'] = logits
        endpoints['logits'] = logits

    return logits, endpoints


def model_fn(features, labels, mode, params=None, config=None):
    """Model function for CNN."""
    data = features['data']
    key = features['key']

    params = params or {}
    learning_rate = params.get('learning_rate', 1e-3)

    arch_kwargs = {
        'n_classes': params.get('n_classes', None),
        'data_format': params.get('data_format', None),
    }
    arch_kwargs = {k: v for k, v in arch_kwargs.items() if v is not None}

    logits, _ = architecture(data, mode, **arch_kwargs)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        'key': key,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    metrics = {'accuracy': accuracy}

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_or_create_global_step())
        # Log accuracy and loss
        with tf.name_scope('train_metrics'):
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('loss', loss)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)
