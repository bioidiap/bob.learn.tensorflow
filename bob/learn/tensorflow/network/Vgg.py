#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
VGG16 and VGG19 wrappers
"""

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg
import tensorflow.contrib.slim as slim


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
           mode=tf.estimator.ModeKeys.TRAIN, **kwargs):
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

    with slim.arg_scope(
        [slim.conv2d],
            trainable=mode==tf.estimator.ModeKeys.TRAIN):

        return vgg.vgg_16(inputs, spatial_squeeze=False)

