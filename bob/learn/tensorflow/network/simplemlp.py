#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavle Korshunov <pavel.korshunov@idiap.ch>
# @date: Fri 15 Sep 2017 13:22 CEST

"""
Simple MLP network architecture.
"""

import tensorflow as tf

import bob.core
logger = bob.core.log.setup("bob.project.savi")

slim = tf.contrib.slim


def get_first_frame_from_timeseries(time_series_inputs, num_time_steps):
    """
    Args:
        time_series_inputs:  Expected to have shape (batch_size, num_time_steps, input_vector_size)

    Returns: Only the first frame of size (batch_size, input_vector_size)

    """
    from tensorflow.python.framework import ops
    # shape inputs correctly
    inputs = ops.convert_to_tensor(time_series_inputs)
    shape = inputs.get_shape().as_list()
    logger.info("MLP: the shape of the inputs: {0}".format(shape))

    input_time_steps = shape[1]  # second dimension must be the number of time steps in LSTM

    if len(shape) == 4:  # when inputs shape is 4, the last dimension must be 1
        if shape[-1] == 1:  # we accept last dimension to be 1, then we just reshape it
            inputs = tf.reshape(inputs, shape=(-1, shape[1], shape[2]))
            logger.info("LSTM: after reshape, the shape of the inputs: {0}".format(inputs.get_shape().as_list()))
        else:
            raise ValueError('The shape of input must be either (batch_size, num_time_steps, input_vector_size) or '
                             '(batch_size, num_time_steps, input_vector_size, 1), but it is {}'.format(shape))

    if input_time_steps % num_time_steps:
        raise ValueError('number of rows in one batch of input ({}) should be '
                         'the same as the num_time_steps of MLP ({})'
                         .format(input_time_steps, num_time_steps))

    # convert inputs into the num_time_steps list of the inputs each of shape (batch_size, input_vector_size)
    list_inputs = tf.unstack(inputs, num_time_steps, 1)

    return list_inputs[0]


def mlp_network(train_data_shuffler, hidden_layer_size=64, num_time_steps=28, num_classes=10, seed=10, reuse=False):
    """
    :param train_data_shuffler: The input is expected to have shape (batch_size, num_time_steps, input_vector_size),
    but only first (batch_size, input_vector_size) will be used as the input input MLP.

    """
    if isinstance(train_data_shuffler, tf.Tensor):
        inputs = train_data_shuffler
    else:
        inputs = train_data_shuffler("data", from_queue=False)

    inputs = get_first_frame_from_timeseries(inputs, num_time_steps)

    initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=seed)
    regularizer = None

    # we just take the first input of size (batch_size, input_vector_size) from the list
    # MLP is 2 fully-connected layers output to the classes
    graph = slim.fully_connected(inputs, hidden_layer_size, activation_fn=tf.nn.relu, scope='fc0',
                                 weights_initializer=initializer, weights_regularizer=regularizer, reuse=reuse)

    graph = slim.fully_connected(graph, num_classes, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, weights_regularizer=regularizer, reuse=reuse)

    return graph
