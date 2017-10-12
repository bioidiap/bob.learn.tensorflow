#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf

def dummy(conv1_kernel_size=3, conv1_output=1, fc1_output=2, seed=10):
    """
    Create all the necessary variables for this CNN

    **Parameters**
        conv1_kernel_size:
        conv1_output:
        fc1_output:
        seed = 10
    """

    slim = tf.contrib.slim

    end_points = dict()
    
    initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=seed)

    graph = slim.conv2d(inputs, conv1_output, conv1_kernel_size, activation_fn=tf.nn.relu,
                        stride=1,
                        weights_initializer=initializer,
                        scope='conv1')
    end_points['conv1'] = graph                            

    graph = slim.flatten(graph, scope='flatten1')
    end_points['flatten1'] = graph        

    graph = slim.fully_connected(graph, fc1_output,
                                 weights_initializer=initializer,
                                 activation_fn=None,
                                 scope='fc1')
    end_points['fc1'] = graph

    return graph, end_points

