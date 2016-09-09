#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer
from operator import mul
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class FullyConnected(Layer):

    """
    2D Convolution
    """

    def __init__(self, name,
                 output_dim,
                 activation=None,
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 use_gpu=False,
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

        super(FullyConnected, self).__init__(name=name,
                                     activation=activation,
                                     weights_initialization=weights_initialization,
                                     bias_initialization=bias_initialization,
                                     use_gpu=use_gpu)

        self.output_dim = output_dim
        self.W = None
        self.b = None
        self.shape = None

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        if self.W is None:
            input_dim = reduce(mul, self.input_layer.get_shape().as_list()[1:])
            self.W = self.weights_initialization(shape=[input_dim, self.output_dim],
                                                 name="W_" + str(self.name))
            # if self.activation is not None:
            self.b = self.bias_initialization(shape=[self.output_dim],
                                              name="b_" + str(self.name))

    def get_graph(self):

        with tf.name_scope(str(self.name)):

            if len(self.input_layer.get_shape()) == 4:
                shape = self.input_layer.get_shape().as_list()
                fc = tf.reshape(self.input_layer, [shape[0], shape[1] * shape[2] * shape[3]])
            else:
                fc = self.input_layer

            if self.activation is not None:
                non_linear_fc = self.activation(tf.matmul(fc, self.W) + self.b)
                output = non_linear_fc
            else:
                output = tf.matmul(fc, self.W) + self.b

            return output
