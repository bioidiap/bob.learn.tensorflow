#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Dummy architecture
"""

import tensorflow as tf


class Dummy(object):

    def __init__(self,
                 conv1_kernel_size=3,
                 conv1_output=1,

                 fc1_output=2,
                 seed=10,
                 n_classes=None):
        """
        Create all the necessary variables for this CNN

        **Parameters**
            conv1_kernel_size=3,
            conv1_output=2,

            n_classes=10

            seed = 10
        """
        self.conv1_output = conv1_output
        self.conv1_kernel_size = conv1_kernel_size
        self.fc1_output = fc1_output
        self.seed = seed
        self.n_classes = n_classes

    def __call__(self, inputs, reuse=False, end_point="logits"):
        slim = tf.contrib.slim

        end_points = dict()
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=self.seed)

        graph = slim.conv2d(inputs, self.conv1_output, self.conv1_kernel_size, activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='conv1')
        end_points['conv1'] = graph                            

        graph = slim.flatten(graph, scope='flatten1')
        end_points['flatten1'] = graph        

        graph = slim.fully_connected(graph, self.fc1_output,
                                     weights_initializer=initializer,
                                     activation_fn=None,
                                     scope='fc1')
        end_points['fc1'] = graph                                     
                                         
        if self.n_classes is not None:
            # Appending the logits layer
            graph = append_logits(graph, self.n_classes, reuse)
            end_points['logits'] = graph

        return end_points[end_point]
