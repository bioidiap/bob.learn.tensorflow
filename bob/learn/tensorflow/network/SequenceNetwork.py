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

    def __init__(self, default_feature_layer=None):
        """
        Base constructor

        **Parameters**
        feature_layer:
        """

        self.sequence_net = OrderedDict()
        self.default_feature_layer = default_feature_layer
        self.input_divide = 1.
        self.input_subtract = 0.
        #self.saver = None

    def add(self, layer):
        """
        Add a layer in the sequence network

        """
        if not isinstance(layer, Layer):
            raise ValueError("Input `layer` must be an instance of `bob.learn.tensorflow.layers.Layer`")
        self.sequence_net[layer.name] = layer

    def compute_graph(self, input_data, feature_layer=None):
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

            if feature_layer is not None and k == feature_layer:
                return input_offset

        return input_offset

    def compute_projection_graph(self, placeholder):
        return self.compute_graph(placeholder)

    def __call__(self, data, session=None, feature_layer=None):

        if session is None:
            session = tf.Session()

        batch_size = data.shape[0]
        width = data.shape[1]
        height = data.shape[2]
        channels = data.shape[3]

        # Feeding the placeholder
        feature_placeholder = tf.placeholder(tf.float32, shape=(batch_size, width, height, channels), name="feature")
        feed_dict = {feature_placeholder: data}

        if feature_layer is None:
            feature_layer = self.default_feature_layer

        return session.run([self.compute_graph(feature_placeholder, feature_layer)], feed_dict=feed_dict)[0]

    def dump_variables(self):

        variables = {}
        for k in self.sequence_net:
            # TODO: IT IS NOT SMART TESTING ALONG THIS PAGE
            if not isinstance(self.sequence_net[k], MaxPooling):
                variables[self.sequence_net[k].W.name] = self.sequence_net[k].W
                variables[self.sequence_net[k].b.name] = self.sequence_net[k].b

        return variables

    def save(self, hdf5, step=None):
        """
        Save the state of the network in HDF5 format
        """

        # Directory that stores the tensorflow variables
        hdf5.create_group('/tensor_flow')
        hdf5.cd('/tensor_flow')

        if step is not None:
            group_name = '/step_{0}'.format(step)
            hdf5.create_group(group_name)
            hdf5.cd(group_name)

        # Iterating the variables of the model
        for v in self.dump_variables().keys():
            hdf5.set(v, self.dump_variables()[v].eval())

        hdf5.cd('..')
        if step is not None:
            hdf5.cd('..')

        hdf5.set('input_divide', self.input_divide)
        hdf5.set('input_subtract', self.input_subtract)

    def load(self, hdf5, shape, session=None):

        if session is None:
            session = tf.Session()

        # Loading the normalization parameters
        self.input_divide = hdf5.read('input_divide')
        self.input_subtract = hdf5.read('input_subtract')

        # Loading variables
        place_holder = tf.placeholder(tf.float32, shape=shape, name="load")
        self.compute_graph(place_holder)
        tf.initialize_all_variables().run(session=session)

        hdf5.cd('/tensor_flow')
        for k in self.sequence_net:
            # TODO: IT IS NOT SMART TESTING ALONG THIS PAGE
            if not isinstance(self.sequence_net[k], MaxPooling):
                #self.sequence_net[k].W.assign(hdf5.read(self.sequence_net[k].W.name))
                self.sequence_net[k].W.assign(hdf5.read(self.sequence_net[k].W.name)).eval(session=session)
                session.run(self.sequence_net[k].W)
                self.sequence_net[k].b.assign(hdf5.read(self.sequence_net[k].b.name)).eval(session=session)
                session.run(self.sequence_net[k].b)


        #if self.saver is None:
        #    variables = self.dump_variables()
        #    variables['input_divide'] = self.input_divide
        #    variables['input_subtract'] = self.input_subtract
        #    self.saver = tf.train.Saver(variables)
        #self.saver.restore(session, path)



    """
    def save(self, session, path, step=None):

        if self.saver is None:
            variables = self.dump_variables()
            variables['mean'] = tf.Variable(10.0)
            #import ipdb; ipdb.set_trace()

            tf.initialize_all_variables().run()
            self.saver = tf.train.Saver(variables)

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
            variables = self.dump_variables()
            variables['input_divide'] = self.input_divide
            variables['input_subtract'] = self.input_subtract
            self.saver = tf.train.Saver(variables)

        self.saver.restore(session, path)
    """
