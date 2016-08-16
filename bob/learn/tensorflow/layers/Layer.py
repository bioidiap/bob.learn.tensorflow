#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf


class Layer(object):

    """
    Layer base class
    """

    def __init__(self, name, activation=None, initialization='xavier', use_gpu=False, seed=10):
        """
        Base constructor

        **Parameters**
          name: Name of the layer
          activation: Tensorflow activation operation (https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html)
          initialization: STILL TO BE DONE.
          use_gpu: I think this is not necessary to explain
          seed: Initialization seed set in Tensor flow
        """
        self.name = name
        self.initialization = initialization
        self.use_gpu = use_gpu
        self.seed = seed

        self.input_layer = None
        self.activation = activation

    def create_variables(self, input_layer):
        NotImplementedError("Please implement this function in derived classes")

    def get_graph(self):
        NotImplementedError("Please implement this function in derived classes")
