#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.learn.tensorflow.layers import maxout
from .utils import append_logits

def light_cnn9(inputs, seed=10, reuse=False):
    """Creates the graph for the Light CNN-9 in 

       Wu, Xiang, et al. "A light CNN for deep face representation with noisy labels." arXiv preprint arXiv:1511.02683 (2015).
    """
    slim = tf.contrib.slim

    with tf.variable_scope('LightCNN9', reuse=reuse):

        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=seed)
        end_points = dict()
                    
        graph = slim.conv2d(inputs, 96, [5, 5], activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='Conv1',
                            reuse=reuse)
        end_points['conv1'] = graph
        
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
        end_points['conv2'] = graph
        
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
        end_points['conv3'] = graph                            

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
        end_points['conv4'] = graph

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
        end_points['conv5'] = graph

        graph = maxout(graph,
                       num_units=128,
                       name='Maxout5')                       

        graph = slim.max_pool2d(graph, [2, 2], stride=2, padding="SAME", scope='Pool4')

        graph = slim.flatten(graph, scope='flatten1')
        end_points['flatten1'] = graph        

        graph = slim.dropout(graph, keep_prob=0.5, scope='dropout1')

        prelogits = slim.fully_connected(graph, 512,
                                     weights_initializer=initializer,
                                     activation_fn=tf.nn.relu,
                                     scope='fc1',
                                     reuse=reuse)
        end_points['fc1'] = prelogits

    return prelogits, end_points

