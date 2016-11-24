#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the lenet architecture
"""

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv1D, FullyConnected
from bob.learn.tensorflow.initialization import Uniform


# construct HardTanh activation function
def hard_tanh(x, name=None):
    one = tf.constant(1, dtype=tf.float32)
    neg_one = tf.constant(-1, dtype=tf.float32)
    return tf.minimum(tf.maximum(x, neg_one), one)


class SimpleAudio(SequenceNetwork):

    def __init__(self,
                 conv1_kernel_size=300,
                 conv1_output=20,
                 conv1_stride=100,

                 fc1_output=40,

                 n_classes=2,
                 default_feature_layer="fc2",

                 seed=10,
                 use_gpu=False
                 ):

        super(SimpleAudio, self).__init__(default_feature_layer=default_feature_layer,
                                          use_gpu=use_gpu)

        self.add(Conv1D(name="conv1", kernel_size=conv1_kernel_size,
                       filters=conv1_output,
                       stride=conv1_stride,
                       activation=hard_tanh,
                       weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       use_gpu=use_gpu
                       ))

        self.add(FullyConnected(name="fc1", output_dim=fc1_output,
                               activation=hard_tanh,
                               weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                               bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                               use_gpu=use_gpu
                               ))

        self.add(FullyConnected(name="fc2", output_dim=n_classes,
                               activation=None,
                               weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                               bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                               use_gpu=use_gpu
                               ))