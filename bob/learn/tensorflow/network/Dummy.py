#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf


def dummy(inputs, reuse=False, is_training_mode = True, trainable_variables=True):
    """
    Create all the necessary variables for this CNN

    **Parameters**
        inputs:
        
        reuse:
    """

    slim = tf.contrib.slim
    end_points = dict()
    
    with tf.variable_scope('Dummy', reuse=reuse):
    
        initializer = tf.contrib.layers.xavier_initializer()
        
        graph = slim.conv2d(inputs, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                            weights_initializer=initializer,
                            trainable=trainable_variables)
        end_points['conv1'] = graph                            
                                
        graph = slim.max_pool2d(graph, [4, 4], scope='pool1')    
        end_points['pool1'] = graph
        
        graph = slim.flatten(graph, scope='flatten1')
        end_points['flatten1'] = graph        

        graph = slim.fully_connected(graph, 50,
                                     weights_initializer=initializer,
                                     activation_fn=None,
                                     scope='fc1',
                                     trainable=trainable_variables)
        end_points['fc1'] = graph


    return graph, end_points

