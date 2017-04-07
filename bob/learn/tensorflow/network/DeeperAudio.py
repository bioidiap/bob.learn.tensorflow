#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Thu 06 April 2017 09:39:36 CEST

"""
Class that creates the lenet architecture
"""

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv1D, FullyConnected, LogSoftMax, MaxPooling
from bob.learn.tensorflow.initialization import Uniform


# construct HardTanh activation function
def hard_tanh(x, name=None):
    one = tf.constant(1, dtype=tf.float32)
    neg_one = tf.constant(-1, dtype=tf.float32)
    return tf.minimum(tf.maximum(x, neg_one), one)


class DeeperAudio(SequenceNetwork):

    def __init__(self,
                 conv1_kernel_size=160,
                 conv1_output=32,
                 conv1_stride=20,

                 pooling_shape=[1, 1, 2, 1],
                 pooling_stride=[1, 1, 2, 1],

                 conv2_kernel_size=32,
                 conv2_output=64,
                 conv2_stride=2,

                 conv3_kernel_size=1,
                 conv3_output=64,
                 conv3_stride=1,

                 fc1_output=60,

                 n_classes=2,
                 default_feature_layer="fc2",

                 seed=10,
                 use_gpu=False
                 ):

        super(DeeperAudio, self).__init__(default_feature_layer=default_feature_layer,
                                          use_gpu=use_gpu)

        self.add(Conv1D(name="conv1", kernel_size=conv1_kernel_size,
                       filters=conv1_output,
                       stride=conv1_stride,
                       activation=hard_tanh,
                       weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       use_gpu=use_gpu
                       ))

        self.add(MaxPooling(name="pooling1", shape=pooling_shape))

        self.add(Conv1D(name="conv2", kernel_size=conv2_kernel_size,
                       filters=conv2_output,
                       stride=conv2_stride,
                       activation=hard_tanh,
                       weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       use_gpu=use_gpu
                       ))

        self.add(MaxPooling(name="pooling2", shape=pooling_shape))


        self.add(Conv1D(name="conv3", kernel_size=conv3_kernel_size,
                       filters=conv3_output,
                       stride=conv3_stride,
                       activation=hard_tanh,
                       weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
                       use_gpu=use_gpu
                       ))

        self.add(MaxPooling(name="pooling3", shape=pooling_shape))

#        self.add(FullyConnected(name="fc1", output_dim=fc1_output,
#                               activation=hard_tanh,
#                               weights_initialization=Uniform(seed=seed, use_gpu=use_gpu),
#                               bias_initialization=Uniform(seed=seed, use_gpu=use_gpu),
#                               use_gpu=use_gpu
#                               ))

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

        self.add(LogSoftMax(name="logsoftmax", activation=None, use_gpu=use_gpu))
