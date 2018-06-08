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


def create_conv_layer(inputs,
                      mode,
                      data_format,
                      endpoints,
                      number,
                      filters,
                      kernel_size,
                      pool_size,
                      pool_strides,
                      skip_pool=False):
    bn_axis = 1 if data_format.lower() == 'channels_first' else 3
    training = mode == tf.estimator.ModeKeys.TRAIN

    name = 'Conv-{}'.format(number)
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        data_format=data_format,
        name=name)
    endpoints[name] = conv

    name = 'BN-{}'.format(number)
    bn = tf.layers.batch_normalization(
        conv, axis=bn_axis, training=training, name=name)
    endpoints[name] = bn

    name = 'Activation-{}'.format(number)
    bn_act = tf.nn.relu(bn, name=name)
    endpoints[name] = bn_act

    name = 'MaxPooling-{}'.format(number)
    if skip_pool:
        pool = bn_act
    else:
        pool = tf.layers.max_pooling2d(
            inputs=bn_act,
            pool_size=pool_size,
            strides=pool_strides,
            padding='same',
            data_format=data_format,
            name=name)
    endpoints[name] = pool

    return pool


def create_dense_layer(inputs, mode, endpoints, number, units):
    training = mode == tf.estimator.ModeKeys.TRAIN

    name = 'FC-{}'.format(number)
    fc = tf.layers.dense(
        inputs=inputs, units=units, activation=None, name=name)
    endpoints[name] = fc

    name = 'BN-{}'.format(number + 5)
    bn = tf.layers.batch_normalization(
        fc, axis=1, training=training, name=name)
    endpoints[name] = bn

    name = 'Activation-{}'.format(number + 5)
    bn_act = tf.nn.relu(bn, name=name)
    endpoints[name] = bn_act

    return bn_act


def base_architecture(input_layer,
                      mode,
                      data_format,
                      skip_first_two_pool=False,
                      **kwargs):
    training = mode == tf.estimator.ModeKeys.TRAIN
    # Keep track of all the endpoints
    endpoints = {}

    # ======================
    # Convolutional Layer #1
    pool1 = create_conv_layer(
        inputs=input_layer,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=1,
        filters=50,
        kernel_size=(5, 5),
        pool_size=(2, 2),
        pool_strides=2,
        skip_pool=skip_first_two_pool)

    # ======================
    # Convolutional Layer #2
    pool2 = create_conv_layer(
        inputs=pool1,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=2,
        filters=100,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        pool_strides=2,
        skip_pool=skip_first_two_pool)

    # ======================
    # Convolutional Layer #3
    pool3 = create_conv_layer(
        inputs=pool2,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=3,
        filters=150,
        kernel_size=(3, 3),
        pool_size=(3, 3),
        pool_strides=2)

    # ======================
    # Convolutional Layer #4
    pool4 = create_conv_layer(
        inputs=pool3,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=4,
        filters=200,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        pool_strides=2)

    # ======================
    # Convolutional Layer #5
    pool5 = create_conv_layer(
        inputs=pool4,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=5,
        filters=250,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        pool_strides=2)

    # ========================
    # Flatten tensor into a batch of vectors
    name = 'MaxPooling-5-Flat'
    pool5_flat = tf.layers.flatten(pool5, name=name)
    endpoints[name] = pool5_flat

    # ========================
    # Fully Connected Layer #1
    fc1 = create_dense_layer(
        inputs=pool5_flat,
        mode=mode,
        endpoints=endpoints,
        number=1,
        units=1000)

    # ========================
    # Dropout
    name = 'dropout'
    dropout = tf.layers.dropout(
        inputs=fc1, rate=0.5, training=training, name=name)
    endpoints[name] = dropout

    # ========================
    # Fully Connected Layer #2
    fc2 = create_dense_layer(
        inputs=dropout, mode=mode, endpoints=endpoints, number=2, units=400)

    return fc2, endpoints


def architecture(input_layer,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 skip_first_two_pool=False,
                 n_classes=2,
                 data_format='channels_last',
                 reuse=False,
                 regularizer=None,
                 **kwargs):

    with tf.variable_scope('PatchCNN', reuse=reuse, regularizer=regularizer):

        fc2, endpoints = base_architecture(
            input_layer=input_layer,
            mode=mode,
            data_format=data_format,
            skip_first_two_pool=skip_first_two_pool)
        # Logits layer
        logits = tf.layers.dense(inputs=fc2, units=n_classes)
        endpoints['FC-3'] = logits
        endpoints['logits'] = logits

    return logits, endpoints


def model_fn(features, labels, mode, params=None, config=None):
    """Model function for CNN."""
    data = features['data']
    key = features['key']

    params = params or {}
    params = {k: v for k, v in params.items() if v is not None}

    initial_learning_rate = params.get('learning_rate', 1e-3)
    momentum = params.get('momentum', 0.99)
    decay_steps = params.get('decay_steps', 1e5)
    decay_rate = params.get('decay_rate', 1e-4)
    staircase = params.get('staircase', True)
    regularization_rate = params.get('regularization_rate', 0)

    arch_kwargs = {
        'skip_first_two_pool': params.get('skip_first_two_pool'),
        'n_classes': params.get('n_classes'),
        'data_format': params.get('data_format'),
        'regularizer': params.get('regularizer')
    }
    arch_kwargs = {k: v for k, v in arch_kwargs.items() if v is not None}

    logits, _ = architecture(data, mode=mode, **arch_kwargs)

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
    # Add the regularization terms to the loss
    if regularization_rate:
        loss += regularization_rate * \
            tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    metrics = {'accuracy': accuracy}

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)

        # for batch normalization to be updated as well:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=global_step)

        # Log accuracy and loss
        with tf.name_scope('train_metrics'):
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning_rate', learning_rate)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)
