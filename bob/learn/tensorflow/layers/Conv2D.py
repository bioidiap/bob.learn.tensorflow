#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class Conv2D(Layer):

    """
    2D Convolution

    **Parameters**

    name: str
      The name of the layer

    activation:
     Tensor Flow activation

    kernel_size: int
      Size of the convolutional kernel

    filters: int
      Number of filters (i.e. numbers of feature maps)

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
                 kernel_size=3,
                 filters=8,
                 stride=[1, 1, 1, 1],
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 batch_norm=False,
                 use_gpu=False,
                 verbosity_level=2,
                 ):
        super(Conv2D, self).__init__(name=name,
                                     activation=activation,
                                     weights_initialization=weights_initialization,
                                     bias_initialization=bias_initialization,
                                     batch_norm=batch_norm,
                                     use_gpu=use_gpu,
                                     )
        self.name = name
        self.kernel_size = kernel_size
        self.filters = filters
        self.W = None
        self.b = None
        self.stride = stride

        bob.core.log.set_verbosity_level(logger, verbosity_level)

        logger.info("+ adding a Conv2D layer ({0}) +".format(name))
        logger.info("\t kernel size = {0}".format(kernel_size))
        logger.info("\t stride = [{0}, {1}]".format(stride[1], stride[2]))
        logger.info("\t number of output features maps = {0}".format(filters))


    def create_variables(self, input_layer, scope="net"):
      
      # get the scope as "network_name/layer_name"
      if scope is not None:
        scope = scope + '/' + self.name
      else:
        scope = self.name
     
      self.scope = scope
      self.input_layer = input_layer

      # TODO: Do an assert here
      if len(input_layer.get_shape().as_list()) != 4:
        raise ValueError("The input as a convolutional layer must have 4 dimensions, "
                         "but {0} were provided".format(len(input_layer.get_shape().as_list())))
      n_channels = input_layer.get_shape().as_list()[3]

      if self.W is None:

        variable = "weights"
        if self.get_varible_by_name(variable) is not None:
          self.W = self.get_varible_by_name(variable)
        else:
          self.W = self.weights_initialization(shape=[self.kernel_size, self.kernel_size, n_channels, self.filters],
                                               name=variable,
                                               scope=scope
                                               )

        variable = "biases"
        if self.get_varible_by_name(variable) is not None:
          self.b = self.get_varible_by_name(variable)
        else:
          self.b = self.bias_initialization(shape=[self.filters],
                                            name=variable,
                                            scope=scope)
      
        logger.info("== Creating variables in layer {0}".format(self.name))
        logger.info("scope --> {0}".format(scope))
        logger.info("weights {0}".format([self.kernel_size, self.kernel_size, n_channels, self.filters]))
        logger.info("biases {0}".format(self.filters))


    def get_graph(self, training_phase=True):

        with tf.name_scope(str(self.name)):
            conv2d = tf.nn.conv2d(self.input_layer, self.W, strides=self.stride, padding='SAME')

            if self.batch_norm:
                conv2d = self.batch_normalize(conv2d, training_phase, scope=self.scope)

            if self.activation is not None:
                output = self.activation(tf.nn.bias_add(conv2d, self.b))
            else:
                output = tf.nn.bias_add(conv2d, self.b)

            return output
