#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer
from operator import mul


class Dropout(Layer):

    """
    Dropout

    **Parameters**
     name: str
       The name of the layer

     keep_prob: float
       With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0.

    """

    def __init__(self, name,
                 keep_prob=0.99
                 ):
        super(Dropout, self).__init__(name=name)
        self.keep_prob = keep_prob

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        return

    def get_graph(self, training_phase=True):

        with tf.name_scope(str(self.name)):
            output = tf.nn.dropout(self.input_layer, self.keep_prob, name=self.name)
            return output
