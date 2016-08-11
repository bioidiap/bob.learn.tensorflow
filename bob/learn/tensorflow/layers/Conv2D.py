#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer


class Conv2D(Layer):

    """
    2D Convolution
    """

    def __init__(self, name, activation=None,
                 kernel_size=3,
                 filters=8,
                 initialization='xavier',
                 use_gpu=False,
                 seed=10
                 ):
        """
        Constructor

        **Parameters**
        input: Layer input
        activation: Tensor Flow activation
        kernel_size: Size of the convolutional kernel
        filters: Number of filters
        initialization: Initialization type
        use_gpu: Store data in the GPU
        seed: Seed for the Random number generation
        """
        super(Conv2D, self).__init__(name, activation=activation, initialization='xavier',
                                     use_gpu=use_gpu, seed=seed)
        self.kernel_size = kernel_size
        self.filters = filters
        self.initialization = initialization
        self.W = None
        self.b = None

    def create_variables(self, input):
        self.input = input
        if self.W is None:
            self.W = create_weight_variables([self.kernel_size, self.kernel_size, 1, self.filters],
                                             seed=self.seed, name=str(self.name), use_gpu=self.use_gpu)
            if self.activation is not None:
                self.b = create_bias_variables([self.filters], name=str(self.name) + "bias", use_gpu=self.use_gpu)

    def get_graph(self):
        with tf.name_scope(str(self.name)):
            conv2d = tf.nn.conv2d(self.input, self.W, strides=[1, 1, 1, 1], padding='SAME')

        if self.activation is not None:
            with tf.name_scope(str(self.name) + 'activation'):
                non_linear_conv2d = tf.nn.tanh(tf.nn.bias_add(conv2d, self.b))
                self.output = non_linear_conv2d
        else:
            self.output = conv2d

        return self.output
