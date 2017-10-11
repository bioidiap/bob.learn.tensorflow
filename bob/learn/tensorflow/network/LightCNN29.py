#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import tensorflow as tf
from bob.learn.tensorflow.layers import maxout
from .utils import append_logits

class LightCNN29(object):
    """Creates the graph for the Light CNN-9 in 

       Wu, Xiang, et al. "A light CNN for deep face representation with noisy labels." arXiv preprint arXiv:1511.02683 (2015).
    """
    def __init__(self,
                 seed=10,
                 n_classes=10):

            self.seed = seed
            self.n_classes = n_classes

    def __call__(self, inputs, reuse=False, end_point="logits"):
        slim = tf.contrib.slim

        end_points = dict()
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=self.seed)
                    
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

        #graph = slim.dropout(graph, keep_prob=0.3, scope='dropout1')

        graph = slim.fully_connected(graph, 512,
                                     weights_initializer=initializer,
                                     activation_fn=tf.nn.relu,
                                     scope='fc1',
                                     reuse=reuse)
        end_points['fc1'] = graph                                     
        
        graph = maxout(graph,
                       num_units=256,
                       name='Maxoutfc1')
        
        graph = slim.dropout(graph, keep_prob=0.3, scope='dropout1')

        if self.n_classes is not None:
            # Appending the logits layer
            graph = append_logits(graph, self.n_classes, reuse)
            end_points['logits'] = graph


        return end_points[end_point]
