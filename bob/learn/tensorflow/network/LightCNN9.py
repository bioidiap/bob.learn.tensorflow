#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.learn.tensorflow.layers import maxout
from .utils import is_trainable


def light_cnn9(inputs,
               seed=10,
               reuse=False,
               trainable_variables=None,
               **kwargs):
    """Creates the graph for the Light CNN-9 in 

       Wu, Xiang, et al. "A light CNN for deep face representation with noisy labels." arXiv preprint arXiv:1511.02683 (2015).
    """
    slim = tf.contrib.slim

    with tf.variable_scope('LightCNN9', reuse=reuse):
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, dtype=tf.float32, seed=seed)
        end_points = dict()
        name = "Conv1"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            inputs,
            96, [5, 5],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)
        end_points[name] = graph

        graph = maxout(graph, num_units=48, name='Maxout1')

        graph = slim.max_pool2d(
            graph, [2, 2], stride=2, padding="SAME", scope='Pool1')

        ####
        name = "Conv2a"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            96, [1, 1],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)

        graph = maxout(graph, num_units=48, name='Maxout2a')

        name = "Conv2"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            192, [3, 3],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)
        end_points[name] = graph

        graph = maxout(graph, num_units=96, name='Maxout2')

        graph = slim.max_pool2d(
            graph, [2, 2], stride=2, padding="SAME", scope='Pool2')

        #####
        name = "Conv3a"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            192, [1, 1],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)

        graph = maxout(graph, num_units=96, name='Maxout3a')

        name = "Conv3"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            384, [3, 3],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)
        end_points[name] = graph

        graph = maxout(graph, num_units=192, name='Maxout3')

        graph = slim.max_pool2d(
            graph, [2, 2], stride=2, padding="SAME", scope='Pool3')

        #####
        name = "Conv4a"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            384, [1, 1],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)

        graph = maxout(graph, num_units=192, name='Maxout4a')

        name = "Conv4"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            256, [3, 3],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)
        end_points[name] = graph

        graph = maxout(graph, num_units=128, name='Maxout4')

        #####
        name = "Conv5a"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            256, [1, 1],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)

        graph = maxout(graph, num_units=128, name='Maxout5a')

        name = "Conv5"
        trainable = is_trainable(name, trainable_variables)
        graph = slim.conv2d(
            graph,
            256, [3, 3],
            activation_fn=tf.nn.relu,
            stride=1,
            weights_initializer=initializer,
            scope=name,
            trainable=trainable,
            reuse=reuse)
        end_points[name] = graph

        graph = maxout(graph, num_units=128, name='Maxout5')

        graph = slim.max_pool2d(
            graph, [2, 2], stride=2, padding="SAME", scope='Pool4')

        graph = slim.flatten(graph, scope='flatten1')
        end_points['flatten1'] = graph

        graph = slim.dropout(graph, keep_prob=0.5, scope='dropout1')

        name = "fc1"
        trainable = is_trainable(name, trainable_variables)
        prelogits = slim.fully_connected(
            graph,
            512,
            weights_initializer=initializer,
            activation_fn=tf.nn.relu,
            scope=name,
            trainable=trainable,
            reuse=reuse)
        end_points['fc1'] = prelogits

    return prelogits, end_points
