#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import tensorflow as tf
from .Layer import Layer
import numpy

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class ConditionConcat(Layer):
  """
  Layer taking a 1d tensor as input and concatenate it with a conditional vector 

  **Parameters**

  name: str
   The name of the layer

  conditional_dim: int
   The dimension of the conditional one-hot vector

  use_gpu: bool
   Store data in the GPU

  """

  def __init__(self, 
               conditional_dim,
               name,
               ):

    super(ConditionConcat, self).__init__(name=name)

    self.conditional_dim = conditional_dim
    
    logger.info("+ adding a Condition Concatenation layer ({0}) +".format(name))
    logger.info("\t conditional dim = {0}".format(conditional_dim))

  def create_variables(self, input_layer, scope=None):
 
    self.input_layer = input_layer
    logger.info("== registering input in layer {0}".format(self.name))


  def get_graph(self, y):
    
    return tf.concat([self.input_layer, y], 1)

