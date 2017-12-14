#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from .utils import is_trainable


def dummy(inputs,
          reuse=False,
          mode=tf.estimator.ModeKeys.TRAIN,
          trainable_variables=None,
          **kwargs):
    """
    Create all the necessary variables for this CNN

    Parameters
    ----------
        inputs:
        
        reuse:

        mode:

        trainable_variables:

    """

    slim = tf.contrib.slim
    end_points = dict()

    # Here is my choice to shutdown the whole scope
    trainable = is_trainable("Dummy", trainable_variables)
    with tf.variable_scope('Dummy', reuse=reuse):

        initializer = tf.contrib.layers.xavier_initializer()
        name = 'conv1'
        graph = slim.conv2d(
            inputs,
            10, [3, 3],
            activation_fn=tf.nn.relu,
            stride=1,
            scope=name,
            weights_initializer=initializer,
            trainable=trainable)
        end_points[name] = graph

        graph = slim.max_pool2d(graph, [4, 4], scope='pool1')
        end_points['pool1'] = graph

        graph = slim.flatten(graph, scope='flatten1')
        end_points['flatten1'] = graph

        name = 'fc1'
        graph = slim.fully_connected(
            graph,
            50,
            weights_initializer=initializer,
            activation_fn=None,
            scope=name,
            trainable=trainable)
        end_points[name] = graph

    return graph, end_points
