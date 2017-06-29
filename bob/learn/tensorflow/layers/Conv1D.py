#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 09 Nov 2016 13:55:22 CEST

import tensorflow as tf
from .Layer import Layer
#from bob.learn.tensorflow.initialization import Xavier
#from bob.learn.tensorflow.initialization import Constant


class Conv1D(Layer):

    """
    1D Convolution
    **Parameters**

    name: str
      The name of the layer

    activation:
     Tensor Flow activation

    kernel_size: int
      Size of the convolutional kernel

    filters: int
      Number of filters

    stride:
      Shape of the stride

    weights_initialization: py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    bias_initialization: py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    batch_norm: bool
      Do batch norm?

    use_gpu: bool
      Store data in the GPU

    """

    def __init__(self, name, activation=None,
                 kernel_size=300,
                 filters=20,
                 stride=100,
                 weights_initialization=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=10),
                 init_value=None,
                 bias_initialization=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=10),
                 batch_norm=False,
                 use_gpu=False
                 ):
        super(Conv1D, self).__init__(name=name,
                                     activation=activation,
                                     weights_initialization=weights_initialization,
                                     bias_initialization=bias_initialization,
                                     batch_norm=batch_norm,
                                     use_gpu=use_gpu,
                                     )
        self.kernel_size = kernel_size
        self.filters = filters
        self.W = None
        self.b = None
        self.stride = stride
        self.init_value = init_value

    def create_variables(self, input_layer):
        self.input_layer = input_layer

        # TODO: Do an assert here
        if len(input_layer.get_shape().as_list()) != 3:
            raise ValueError("The input as a convolutional layer must have 3 dimensions, "
                             "but {0} were provided".format(len(input_layer.get_shape().as_list())))
        n_channels = input_layer.get_shape().as_list()[2]

        if self.W is None:
            if self.init_value is None:
                self.init_value = self.kernel_size * n_channels
            self.W = self.weights_initialization(shape=[self.kernel_size, n_channels, self.filters],
                                                 name="w_" + str(self.name),
                                                 scope="w_" + str(self.name),
                                                 init_value=self.init_value
                                                 )

            self.b = self.bias_initialization(shape=[self.filters],
                                              name="b_" + str(self.name) + "bias",
                                              scope="b_" + str(self.name),
                                              init_value=self.init_value
                                              )

    def get_graph(self, training_phase=True):

        with tf.name_scope(str(self.name)):
            conv1d = tf.nn.conv1d(self.input_layer, self.W, stride=self.stride, padding='VALID')

            if self.batch_norm:
                conv1d = self.batch_normalize(conv1d, training_phase)

            if self.activation is not None:
                output = self.activation(tf.nn.bias_add(conv1d, self.b))
            else:
                output = tf.nn.bias_add(conv1d, self.b)

            return output
