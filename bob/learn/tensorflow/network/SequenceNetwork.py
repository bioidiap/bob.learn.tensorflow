#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 11 Aug 2016 09:39:36 CEST

"""
Class that creates the lenet architecture
"""

from ..util import *
import tensorflow as tf
import abc
import six

from collections import OrderedDict
from bob.learn.tensorflow.layers import Layer


class SequenceNetwork(six.with_metaclass(abc.ABCMeta, object)):
    """
    Base class to create architectures using TensorFlow
    """

    def __init__(self, feature_layer=None):
        """
        Base constructor

        **Parameters**
        feature_layer:
        """

        self.sequence_net = OrderedDict()
        self.feature_layer = feature_layer

    def add(self, layer):
        """
        Add a layer in the sequence network

        """
        if not isinstance(layer, Layer):
            raise ValueError("Input `layer` must be an instance of `bob.learn.tensorflow.layers.Layer`")
        self.sequence_net[layer.name] = layer

    def compute_graph(self, input_data, cut=False):
        """
        Given the current network, return the Tensorflow graph

         **Parameter**
          input_data:
          cut:
        """

        input_offset = input_data
        for k in self.sequence_net.keys():
            current_layer = self.sequence_net[k]
            current_layer.create_variables(input_offset)
            input_offset = current_layer.get_graph()

            if cut and k == self.feature_layer:
                return input_offset

        return input_offset

    def compute_projection_graph(self, placeholder):
        return self.compute_graph(placeholder, cut=True)

    def __call__(self, feed_dict, session):
        #placeholder
        return session.run([self.graph], feed_dict=feed_dict)[0]
