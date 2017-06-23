#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import tensorflow as tf
from .Layer import Layer
import numpy

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class ImToCondFeatureMap(Layer):
  """
  Layer taking an image (tensor) as input and output the conditional feature map
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

  use_gpu: bool
   Store data in the GPU

  """

  def __init__(self, conditional_dim,
               name,
               ):

    super(ImToCondFeatureMap, self).__init__(
                                             name=name,
                                             activation=None,
                                             weights_initialization=None,
                                             bias_initialization=None,
                                             batch_norm=False,
                                             use_gpu=False
                                            )

    logger.info("+ adding a Concatenation layer ({0}) +".format(name))
    logger.info("\t conditional dimension = {0}".format(conditional_dim))

  def create_variables(self, input_layer, scope=None):
    self.input_layer = input_layer

  def get_graph(self, y):

    """

    **Parameters**

    name: y
    The conditioning vector (one-hot encoded)

    """ 
    # concatenate them to the image
    x_shapes = input_layer.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([input_layer, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

