#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer
from operator import mul
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant
import numpy

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class FullyConnected(Layer):
    """
    Fully Connected layer

    **Parameters**

    name: str
      The name of the layer

    output_dim: int
      Size of the output

    activation:
      Tensor Flow activation

    weights_initialization:  py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    bias_initialization:  py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    batch_norm: bool
      Do batch norm?

    use_gpu: bool
      Store data in the GPU

    """

    def __init__(self, name,
                 output_dim,
                 activation=None,
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 batch_norm=False,
                 init_value=None,
                 use_gpu=False,
                 ):

        super(FullyConnected, self).__init__(name=name,
                                             activation=activation,
                                             weights_initialization=weights_initialization,
                                             bias_initialization=bias_initialization,
                                             batch_norm=batch_norm,
                                             use_gpu=use_gpu
                                             )

        self.output_dim = output_dim
        self.W = None
        self.b = None
        self.shape = None
        self.init_value = init_value
        self.name = name
        
        logger.info("+ adding a Fully Connected layer ({0}) +".format(name))


    def create_variables(self, input_layer, scope="net"):

      # get the scope as "network_name/layer_name"
      if scope is not None:
        scope = scope + '/' + self.name
      else:
        scope = self.name
     
      self.scope = scope
      self.input_layer = input_layer
      
      if self.W is None:
        input_dim = numpy.prod(self.input_layer.get_shape().as_list()[1:])
        if self.init_value is None:
          self.init_value = input_dim

        variable = "weights"
        if self.get_varible_by_name(variable) is not None:
          self.W = self.get_varible_by_name(variable)
        else:
          self.W = self.weights_initialization(shape=[input_dim, self.output_dim],
                                               name=variable,
                                               scope=scope,
                                               init_value=self.init_value
                                               )
            
        variable = "biases"
        if self.get_varible_by_name(variable) is not None:
          self.b = self.get_varible_by_name(variable)
        else:
          self.b = self.bias_initialization(shape=[self.output_dim],
                                            name=variable,
                                            scope=scope,
                                            init_value=self.init_value
                                           )
        
        logger.info("== Creating variables in layer {0}".format(self.name))
        logger.info("scope --> {0}".format(scope))
        logger.info("weights {0}".format([input_dim, self.output_dim]))
        logger.info("biases {0}".format(self.output_dim))


    def get_graph(self, training_phase=True):

        with tf.name_scope(str(self.name)):

            if len(self.input_layer.get_shape()) == 4 or len(self.input_layer.get_shape()) == 3:
                shape = self.input_layer.get_shape().as_list()
                fc = tf.reshape(self.input_layer, [-1, numpy.prod(shape[1:])])
            else:
                fc = self.input_layer

            if self.batch_norm:
                fc = self.batch_normalize(fc, training_phase, scope=self.scope)

            if self.activation is not None:
                non_linear_fc = self.activation(tf.matmul(fc, self.W) + self.b)
                output = non_linear_fc
            else:
                output = tf.matmul(fc, self.W) + self.b

            return output
