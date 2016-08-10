#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensoflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer


class MaxPooling(Layer):

    def __init__(self, input, use_gpu=False):
        """
        Constructor
        """
        super(MaxPooling, self).__init__(input, use_gpu=False)

    def get_graph(self):
        tf.nn.max_pool(self.input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
