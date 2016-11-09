#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from bob.learn.tensorflow.util import *
from .MaxPooling import MaxPooling


class AveragePooling(MaxPooling):

    """
    Wraps the tensorflow average pooling

    **Parameters**
     name: The name of the layer
     shape: Shape of the pooling kernel
     stride: Shape of the stride
     batch_norm: Do batch norm?
     activation: Tensor Flow activation
    """

    def __init__(self, name, shape=[1, 2, 2, 1],
                 strides=[1, 1, 1, 1],
                 batch_norm=False,
                 activation=None):

        super(AveragePooling, self).__init__(name, activation=activation, batch_norm=batch_norm)
        self.shape = shape
        self.strides = strides

    def get_graph(self, training_phase=True):
        with tf.name_scope(str(self.name)):
            output = tf.nn.avg_pool(self.input_layer, ksize=self.shape, strides=self.strides, padding='SAME')

            if self.batch_norm:
                output = self.batch_normalize(output, training_phase)

            if self.activation is not None:
                output = self.activation(output)

            return output
