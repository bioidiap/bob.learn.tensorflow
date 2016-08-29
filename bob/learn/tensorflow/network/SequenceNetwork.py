#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 11 Aug 2016 09:39:36 CEST

"""
Class that creates the lenet architecture
"""

import tensorflow as tf
import abc
import six
import os

from collections import OrderedDict
from bob.learn.tensorflow.layers import Layer, MaxPooling


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
        self.saver = None

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
          cut: Name of the layer that you want to cut.
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

    def __call__(self, data, session=None):

        if session is None:
            session = tf.Session()

        batch_size = data.shape[0]
        width = data.shape[1]
        height = data.shape[2]
        channels = data.shape[3]

        # Feeding the placeholder
        feature_placeholder = tf.placeholder(tf.float32, shape=(batch_size, width, height, channels), name="feature")
        feed_dict = {feature_placeholder: data}

        return session.run([self.compute_projection_graph(feature_placeholder)], feed_dict=feed_dict)[0]

    def dump_variables(self):

        variables = {}
        for k in self.sequence_net:
            # TODO: IT IS NOT SMART TESTING ALONG THIS PAGE
            if not isinstance(self.sequence_net[k], MaxPooling):
                variables[self.sequence_net[k].W.name] = self.sequence_net[k].W
                variables[self.sequence_net[k].b.name] = self.sequence_net[k].b

        return variables

    def save(self, session, path, step=None):

        if self.saver is None:
            self.saver = tf.train.Saver(self.dump_variables())

        if step is None:
            return self.saver.save(session, os.path.join(path, "model.ckpt"))
        else:
            return self.saver.save(session, os.path.join(path, "model" + str(step) + ".ckpt"))

    def load(self, path, shape, session=None):

        if session is None:
            session = tf.Session()

        # Loading variables
        place_holder = tf.placeholder(tf.float32, shape=shape, name="load")
        self.compute_graph(place_holder)
        tf.initialize_all_variables().run(session=session)

        if self.saver is None:
            self.saver = tf.train.Saver(self.dump_variables())

        self.saver.restore(session, path)
