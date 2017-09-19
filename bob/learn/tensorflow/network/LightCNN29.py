#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import tensorflow as tf
from bob.learn.tensorflow.layers import maxout

class LightCNN29(object):
    """Creates the graph for the Light CNN-9 in 

       Wu, Xiang, et al. "A light CNN for deep face representation with noisy labels." arXiv preprint arXiv:1511.02683 (2015).
    """
    def __init__(self,
                 seed=10,
                 n_classes=10,
                 device="/cpu:0",
                 batch_norm=False):

            self.seed = seed
            self.device = device
            self.batch_norm = batch_norm
            self.n_classes = n_classes

    def __call__(self, inputs, reuse=False):
        slim = tf.contrib.slim

        #with tf.device(self.device):

        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=self.seed)
                    
        graph = slim.conv2d(inputs, 96, [5, 5], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv1',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=48,
                       name='Maxout1')

        graph = slim.max_pool2d(graph, [2, 2], stride=2, padding="SAME", scope='Pool1')

        ####

        graph = slim.conv2d(graph, 96, [1, 1], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv2a',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=48,
                       name='Maxout2a')

        graph = slim.conv2d(graph, 192, [3, 3], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv2',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=96,
                       name='Maxout2')

        graph = slim.max_pool2d(graph, [2, 2], stride=2, padding="SAME", scope='Pool2')

        #####

        graph = slim.conv2d(graph, 192, [1, 1], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv3a',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=96,
                       name='Maxout3a')

        graph = slim.conv2d(graph, 384, [3, 3], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv3',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=192,
                       name='Maxout3')

        graph = slim.max_pool2d(graph, [2, 2], stride=2, padding="SAME", scope='Pool3')

        #####

        graph = slim.conv2d(graph, 384, [1, 1], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv4a',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=192,
                       name='Maxout4a')

        graph = slim.conv2d(graph, 256, [3, 3], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv4',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=128,
                       name='Maxout4')

        #####

        graph = slim.conv2d(graph, 256, [1, 1], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv5a',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=128,
                       name='Maxout5a')

        graph = slim.conv2d(graph, 256, [3, 3], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv5',
                            reuse=reuse)

        graph = maxout(graph,
                       num_units=128,
                       name='Maxout5')

        graph = slim.max_pool2d(graph, [2, 2], stride=2, padding="SAME", scope='Pool4')

        graph = slim.flatten(graph, scope='flatten1')

        #graph = slim.dropout(graph, keep_prob=0.3, scope='dropout1')

        graph = slim.fully_connected(graph, 512,
                                     weights_initializer=initializer,
                                     activation_fn=tf.nn.relu,
                                     scope='fc1',
                                     reuse=reuse)
        graph = maxout(graph,
                       num_units=256,
                       name='Maxoutfc1')
        
        graph = slim.dropout(graph, keep_prob=0.3, scope='dropout1')

        graph = slim.fully_connected(graph, self.n_classes,
                                     weights_initializer=initializer,
                                     activation_fn=None,
                                     scope='fc2',
                                     reuse=reuse)

        return graph
