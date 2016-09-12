#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer


class MaxPooling(Layer):

    def __init__(self, name):
        """
        Constructor
        """
        super(MaxPooling, self).__init__(name, use_gpu=False)

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        return

    def get_graph(self):
        with tf.name_scope(str(self.name)):
            output = tf.nn.max_pool(self.input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            return output
