#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavle Korshunov <pavel.korshunov@idiap.ch>
# @date: Fri 15 Sep 2017 13:22 CEST

"""
Simple 2-layered 2D CNN network architecture.
"""

import tensorflow as tf

import bob.core
logger = bob.core.log.setup("bob.project.savi")

slim = tf.contrib.slim


def simple2Dcnn_network(train_data_shuffler, num_classes=10, seed=10, reuse=False):
    """
    :param train_data_shuffler: The input is expected to have shape (batch_size, num_time_steps, input_vector_size),

    """
    if isinstance(train_data_shuffler, tf.Tensor):
        inputs = train_data_shuffler
    else:
        inputs = train_data_shuffler("data", from_queue=False)

    initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=seed)
    regularizer = None

    graph = slim.conv2d(inputs, 32, [5, 5], activation_fn=tf.nn.relu,
                        stride=1,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer,
                        scope='Conv1',
                        reuse=reuse)

    graph = slim.max_pool2d(graph, [1, 2], stride=2, padding="SAME", scope='Pool1')

    graph = slim.conv2d(graph, 32, [3, 3], activation_fn=tf.nn.relu,
                        stride=1,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer,
                        scope='Conv2',
                        reuse=reuse)

    graph = slim.max_pool2d(graph, [1, 2], stride=2, padding="SAME", scope='Pool2')

    graph = slim.flatten(graph, scope='flatten1')

    graph = slim.fully_connected(graph, 80,
                                 weights_initializer=initializer,
                                 activation_fn=tf.nn.relu,
                                 scope='fc0',
                                 reuse=reuse)

    graph = slim.fully_connected(graph, num_classes, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, weights_regularizer=regularizer, reuse=reuse)

    return graph
