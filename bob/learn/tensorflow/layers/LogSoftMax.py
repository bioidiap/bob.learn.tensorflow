#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer


class LogSoftMax(Layer):
    """
    Wraps the tensorflow Log_softmax

    **Parameters**

    name: str
      The name of the layer

    stride:
      Shape of the stride

    batch_norm: bool
      Do batch norm?

    activation: bool
      Tensor Flow activation

    """

    def __init__(self, name,
                 batch_norm=False,
                 activation=None,
                 use_gpu=False):
        super(LogSoftMax, self).__init__(name, use_gpu=use_gpu, activation=activation, batch_norm=batch_norm)

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        return

    def get_graph(self, training_phase=True):
        with tf.name_scope(str(self.name)):
            output = tf.nn.log_softmax(self.input_layer)

            if self.batch_norm:
                output = self.batch_normalize(output, training_phase)

            if self.activation is not None:
                output = self.activation(output)

            return output
