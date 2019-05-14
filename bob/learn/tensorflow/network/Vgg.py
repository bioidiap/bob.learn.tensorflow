#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
VGG16 and VGG19 wrappers
"""

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib import layers
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from .utils import is_trainable


def vgg_19(inputs,
           reuse=None,                      
           mode=tf.estimator.ModeKeys.TRAIN, **kwargs):
    """
    Oxford Net VGG 19-Layers version E Example from tf-slim

    https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/vgg.py

    **Parameters**:

        inputs: a 4-D tensor of size [batch_size, height, width, 3].

        reuse: whether or not the network and its variables should be reused. To be
               able to reuse 'scope' must be given.

        mode:
           Estimator mode keys
    """

    with slim.arg_scope(
        [slim.conv2d],
            trainable=mode==tf.estimator.ModeKeys.TRAIN):

        return vgg.vgg_19(inputs, spatial_squeeze=False)


def vgg_16(inputs,
           reuse=None,                      
           mode=tf.estimator.ModeKeys.TRAIN,
           trainable_variables=None,
           scope="vgg_16",
           **kwargs):
    """
    Oxford Net VGG 16-Layers version E Example from tf-slim

    https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/vgg.py

    **Parameters**:

        inputs: a 4-D tensor of size [batch_size, height, width, 3].

        reuse: whether or not the network and its variables should be reused. To be
               able to reuse 'scope' must be given.

        mode:
           Estimator mode keys
    """

    dropout_keep_prob = 0.5
    end_points = {}
    

    with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.

        with slim.arg_scope(
            [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d], outputs_collections=end_points_collection):
    
            with slim.arg_scope(
                [slim.conv2d],
                    trainable=mode==tf.estimator.ModeKeys.TRAIN):

                name = "conv1"
                trainable = is_trainable(name, trainable_variables, mode=mode)
                net = layers_lib.repeat(
                    inputs, 2, layers.conv2d, 64, [3, 3], scope=name, trainable=trainable)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
                end_points[name] = net

                name = "conv2"        
                trainable = is_trainable(name, trainable_variables, mode=mode)
                net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope=name, trainable=trainable)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
                end_points[name] = net        
                
                name = "conv3"
                trainable = is_trainable(name, trainable_variables, mode=mode)
                net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope=name, trainable=trainable)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
                end_points[name] = net        

                name = "conv4"
                trainable = is_trainable(name, trainable_variables, mode=mode)        
                net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope=name, trainable=trainable)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
                end_points[name] = net        

                name = "conv5"
                trainable = is_trainable(name, trainable_variables, mode=mode)
                net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope=name, trainable=trainable)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
                end_points[name] = net        
                
                net = layers.flatten(net)
                
                # Use conv2d instead of fully_connected layers.
                name = "fc6"
                trainable = is_trainable(name, trainable_variables, mode=mode)
                net = layers.fully_connected(net, 4096, scope=name, trainable=trainable)
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=mode==tf.estimator.ModeKeys.TRAIN, scope='dropout6')
                end_points[name] = net            

                name = "fc7"            
                trainable = is_trainable(name, trainable_variables, mode=mode)
                net = layers.fully_connected(net, 4096, scope=name, trainable=trainable)
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=mode==tf.estimator.ModeKeys.TRAIN, scope='dropout7')

                end_points[name] = net            
  
    # Convert end_points_collection into a end_point dict.
    return net, end_points
    #return vgg.vgg_16(inputs, spatial_squeeze=False)

