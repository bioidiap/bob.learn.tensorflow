#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .MaxPooling import MaxPooling

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class AveragePooling(MaxPooling):

    """
    Wraps the tensorflow average pooling

    **Parameters**

    name: str
      The name of the layer

    shape:
      Shape of the pooling kernel

    stride:
      Shape of the stride

    batch_norm: bool
      Do batch norm?

    activation: bool
      Tensor Flow activation

    """

    def __init__(self, name, shape=[1, 2, 2, 1],
                 strides=[1, 1, 1, 1],
                 batch_norm=False,
                 activation=None):

        super(AveragePooling, self).__init__(name, activation=activation, batch_norm=batch_norm)
        self.shape = shape
        self.strides = strides

    def create_variables(self, input_layer, scope=None):
    
      # get the scope as "network_name/layer_name"
      if scope is not None:
        scope = scope + '/' + self.name
      else:
        scope = self.name

      self.scope = scope


      self.input_layer = input_layer
      logger.info("== registering input in Average Pooling layer {0}".format(self.name))


    def get_graph(self, training_phase=True):
        with tf.name_scope(str(self.name)):
            output = tf.nn.avg_pool(self.input_layer, ksize=self.shape, strides=self.strides, padding='SAME')

            if self.batch_norm:
                output = self.batch_normalize(output, training_phase, scope=self.scope)

            if self.activation is not None:
                output = self.activation(output)

            return output
