#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer


class MaxPooling(Layer):
    """
    Wraps the tensorflow max pooling

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
        super(MaxPooling, self).__init__(name, use_gpu=False, activation=activation, batch_norm=batch_norm)
        self.shape = shape
        self.strides = strides

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        return

    def get_graph(self, training_phase=True):
        with tf.name_scope(str(self.name)):
            output = tf.nn.max_pool(self.input_layer, ksize=self.shape, strides=self.strides, padding='SAME')

            if self.batch_norm:
                output = self.batch_normalize(output, training_phase)

            if self.activation is not None:
                output = self.activation(output)

            return output
