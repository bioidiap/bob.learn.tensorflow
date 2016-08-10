#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensoflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer


class Conv2D(Layer):

    """
    2D Convolution
    """

    def __init__(self, input, activation=None,
                 kernel_size=3,
                 filters=8,
                 initialization='xavier',
                 use_gpu=False,
                 seed=10
                 ):
        """
        Base constructor

        **Parameters**
        input: Layer input
        activation: Tensor Flow activation
        kernel_size: Size of the convolutional kernel
        filters: Number of filters
        initialization: Initialization type
        use_gpu: Store data in the GPU
        seed: Seed for the Random number generation
        """
        super(Conv2D, self).__init__(input, initialization='xavier', use_gpu=False, seed=10)
        self.activation = activation

        self.W = create_weight_variables([kernel_size, kernel_size, 1, filters],
                                         seed=seed, name="conv", use_gpu=use_gpu)

        if activation is not None:
            self.b = create_bias_variables([filters], name="bias", use_gpu=self.use_gpu)

    def get_graph(self):
        with tf.name_scope('conv'):
            conv = tf.nn.conv2d(self.input, self.W, strides=[1, 1, 1, 1], padding='SAME')

        with tf.name_scope('activation'):
            non_linearity = tf.nn.tanh(tf.nn.bias_add(conv, self.b))

        return non_linearity
