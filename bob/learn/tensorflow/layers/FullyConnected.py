#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer
from operator import mul


class FullyConnected(Layer):

    """
    2D Convolution
    """

    def __init__(self, name, output_dim, activation=None,
                 initialization='xavier',
                 use_gpu=False,
                 seed=10
                 ):
        """
        Constructor

        **Parameters**
        input: Layer input
        activation: Tensor Flow activation
        initialization: Initialization type
        use_gpu: Store data in the GPU
        seed: Seed for the Random number generation
        """
        super(FullyConnected, self).__init__(name, activation=activation,
                                             initialization=initialization, use_gpu=use_gpu, seed=seed)
        self.output_dim = output_dim
        self.W = None
        self.b = None
        self.shape = None

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        if self.W is None:
            input_dim = reduce(mul, self.input_layer.get_shape().as_list())

            self.W = create_weight_variables([input_dim, self.output_dim],
                                             seed=self.seed, name=str(self.name), use_gpu=self.use_gpu)
            if self.activation is not None:
                self.b = create_bias_variables([self.output_dim], name=str(self.name)+"_bias", use_gpu=self.use_gpu)

    def get_graph(self):
        with tf.name_scope('fc'):

            if len(self.input_layer.get_shape()) == 4:
                shape = self.input_layer.get_shape().as_list()
                fc = tf.reshape(self.input_layer, [shape[0], shape[1] * shape[2] * shape[3]])

        if self.activation is not None:
            with tf.name_scope('activation'):
                non_linear_fc = tf.nn.tanh(tf.matmul(fc, self.W) + self.b)
                output = non_linear_fc
        else:
            output = fc

        return output
