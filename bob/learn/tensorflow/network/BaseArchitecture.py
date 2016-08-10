#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the lenet architecture
"""

from ..util import *
import tensorflow as tf
import abc
import six


class BaseArchitecture(six.with_metaclass(abc.ABCMeta, object)):
    """
    Base class to create architectures using TensorFlow
    """

    def __init__(self, seed=10, use_gpu=False):
        """
        Base constructor
        """
        self.seed = seed
        self.use_gpu = use_gpu
        self.create_variables()

    def create_variables(self):
        """
        Create the Tensor flow variables
        """
        raise NotImplementedError("Please implement this function in derived classes")

    def create_graph(self, data):
        """
        Create the Tensor flow variables

        **Parameters**
            data: Tensorflow Placeholder

        **Returns**
            Network output
        """

        raise NotImplementedError("Please implement this function in derived classes")
