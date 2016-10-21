#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class Conv2D(Layer):

    """
    2D Convolution
    """

    def __init__(self, name, activation=None,
                 kernel_size=3,
                 filters=8,
                 stride=[1, 1, 1, 1],
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 use_gpu=False
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
        super(Conv2D, self).__init__(name=name,
                                     activation=activation,
                                     weights_initialization=weights_initialization,
                                     bias_initialization=bias_initialization,
                                     use_gpu=use_gpu,


                                     )
        self.kernel_size = kernel_size
        self.filters = filters
        self.W = None
        self.b = None
        self.stride = stride

    def create_variables(self, input_layer):
        self.input_layer = input_layer

        # TODO: Do an assert here
        if len(input_layer.get_shape().as_list()) != 4:
            raise ValueError("The input as a convolutional layer must have 4 dimensions, "
                             "but {0} were provided".format(len(input_layer.get_shape().as_list())))
        n_channels = input_layer.get_shape().as_list()[3]

        if self.W is None:
            self.W = self.weights_initialization(shape=[self.kernel_size, self.kernel_size, n_channels, self.filters],
                                                 name="w_" + str(self.name),
                                                 scope="w_" + str(self.name)
                                                 )

            self.b = self.bias_initialization(shape=[self.filters],
                                              name="b_" + str(self.name) + "bias",
                                              scope="b_" + str(self.name)
                                              )

    def get_graph(self):

        with tf.name_scope(str(self.name)):
            conv2d = tf.nn.conv2d(self.input_layer, self.W, strides=self.stride, padding='SAME')

            if self.activation is not None:
                output = self.activation(tf.nn.bias_add(conv2d, self.b))
            else:
                output = tf.nn.bias_add(conv2d, self.b)

            return output
