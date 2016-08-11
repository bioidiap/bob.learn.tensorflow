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
        input: Layer input

        """
        self.name = name
        self.initialization = initialization
        self.use_gpu = use_gpu
        self.seed = seed

        self.input = None
        self.activation = None
        self.output = None

    def create_variables(self, input):
        NotImplementedError("Please implement this function in derived classes")

    def get_graph(self):
        NotImplementedError("Please implement this function in derived classes")

    def get_shape(self):
        if self.output is None:
            NotImplementedError("This class was not implemented properly")
        else:
            return self.output.get_shape()
