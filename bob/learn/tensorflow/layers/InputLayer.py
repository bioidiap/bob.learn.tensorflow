#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer


class InputLayer(Layer):

    def __init__(self, name, input_data, use_gpu=False):
        """
        Constructor
        """
        super(InputLayer, self).__init__(name, use_gpu=False)
        self.original_layer = input_data
        self.__shape = input_data.get_shape()

    def create_variables(self, input_layer):
        return

    def get_graph(self, training_phase=True):
        return self.original_layer
