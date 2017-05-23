#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import tensorflow as tf
from .Layer import Layer
from operator import mul
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant
import numpy

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class FlatTo2D(Layer):
  """
  Layer taking a 1d tensor as input and projecting it to 2D feature maps
  (resulting in a 3d tensor)

  **Parameters**

  name: str
   The name of the layer

  output_width: int
   Width of the output 

  output_height: int
   Height of the output 

  output_depth: int
   Depth of the output (i.e. number of feature maps) 

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
               output_width,
               output_height,
               output_depth,
               activation=None,
               weights_initialization=Xavier(),
               bias_initialization=Constant(),
               batch_norm=False,
               init_value=None,
               use_gpu=False,
               ):

    super(FlatTo2D, self).__init__(name=name,
                                   activation=activation,
                                   weights_initialization=weights_initialization,
                                   bias_initialization=bias_initialization,
                                   batch_norm=batch_norm,
                                   use_gpu=use_gpu
                                   )

    self.output_dim_total = output_width * output_height * output_depth
    self.output_width = output_width
    self.output_height = output_height
    self.output_depth = output_depth
    self.W = None
    self.b = None
    self.shape = None
    self.init_value = init_value
    
    logger.info("+ adding a Projection layer ({0}) +".format(name))
    logger.info("\t output width = {0}".format(output_width))
    logger.info("\t output height = {0}".format(output_height))
    logger.info("\t output depth = {0}".format(output_depth))

  def create_variables(self, input_layer, scope=None):
 
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
       
        # retrieve the weights
        if self.get_varible_by_name(variable) is not None:
          self.W = self.get_varible_by_name(variable)
        else:
          self.W = self.weights_initialization(shape=[input_dim, self.output_dim_total],
                                                     name="weights",
                                                     scope=scope,
                                                     init_value=self.init_value
                                                     )
        
        # retrieve the biases
        variable = "biases"
        if self.get_varible_by_name(variable) is not None:
          self.b = self.get_varible_by_name(variable)
        else:
          self.b = self.bias_initialization(shape=[self.output_dim_total],
                                            name="biases",
                                            scope=scope,
                                            init_value=self.init_value
                                            )
      
        logger.info("== Creating variables in layer {0}".format(self.name))
        logger.info("scope --> {0}".format(scope))
        logger.info("weights {0}x{1}".format(input_dim, self.output_dim_total))
        logger.info("biases {0}".format(self.output_dim_total))


  def get_graph(self, training_phase=True):

    with tf.name_scope(str(self.name)):

      assert self.input_layer.get_shape() > 2, "Shape of the tensor is not flat !"
      fc = self.input_layer
      linear_fc = tf.matmul(fc, self.W) + self.b
        
      # reshape
      cube = tf.reshape(linear_fc, [-1, self.output_width, self.output_height, self.output_depth])

      if self.batch_norm:
        cube = self.batch_normalize(cube, training_phase, scope=self.scope)

      if self.activation is not None:
        output = self.activation(cube)
      else:
        output = cube

      return output
