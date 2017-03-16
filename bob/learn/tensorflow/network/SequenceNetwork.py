#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 11 Aug 2016 09:39:36 CEST


import tensorflow as tf
import abc
import six
import numpy
import pickle

from bob.learn.tensorflow.layers import Layer, MaxPooling, Dropout, Conv2D, FullyConnected
from bob.learn.tensorflow.utils.session import Session


class SequenceNetwork(six.with_metaclass(abc.ABCMeta, object)):
    """
    Sequential model is a linear stack of :py:mod:`bob.learn.tensorflow.layers`.

    **Parameters**

        default_feature_layer: Default layer name (:py:obj:`str`) used as a feature layer.

        use_gpu: If ``True`` uses the GPU in the computation.
    """

    def __init__(self,
                 graph=None,
                 default_feature_layer=None,
                 use_gpu=False):

        self.base_graph = graph
        self.default_feature_layer = default_feature_layer
        self.use_gpu = use_gpu

    def __del__(self):
        tf.reset_default_graph()

    def __call__(self, data, feature_layer=None):
        """Run a graph and compute the embeddings

        **Parameters**

        data: tensorflow placeholder as input data

        session: tensorflow `session <https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#Session>`_

        feature_layer: Name of the :py:class:`bob.learn.tensorflow.layer.Layer` that you want to "cut".
                       If `None` will run the graph until the end.
        """

        session = Session.instance().session

        # Feeding the placeholder
        if self.inference_placeholder is None:
            self.compute_inference_placeholder(data.shape[1:])
        feed_dict = {self.inference_placeholder: data}

        if self.inference_graph is None:
            self.compute_inference_graph(self.inference_placeholder, feature_layer)

        embedding = session.run([self.inference_graph], feed_dict=feed_dict)[0]

        return embedding

    def predict(self, data):
        return numpy.argmax(self(data), 1)

    """
    def variable_summaries(self, var, name):
        #Attach a lot of summaries to a Tensor.
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

    """

    def compute_magic_number(self, hypothetic_image_dimensions=(28, 28, 1)):
        """

        Here it is done an estimative of the capacity of DNN.

        **Parameters**

            hypothetic_image_dimensions: Possible image dimentions `w, h, c` (width x height x channels)
        """

        stride = 1# ALWAYS EQUALS TO ONE
        current_image_dimensions = list(hypothetic_image_dimensions)

        samples_per_sample = 0
        flatten_dimension = numpy.prod(current_image_dimensions)
        for k in self.sequence_net.keys():
            current_layer = self.sequence_net[k]

            if isinstance(current_layer, Conv2D):
                #samples_per_sample += current_layer.filters * current_layer.kernel_size * current_image_dimensions[0] + current_layer.filters
                #samples_per_sample += current_layer.filters * current_layer.kernel_size * current_image_dimensions[1] + current_layer.filters
                samples_per_sample += current_layer.filters * current_image_dimensions[0] * current_image_dimensions[1] + current_layer.filters

                current_image_dimensions[2] = current_layer.filters
                flatten_dimension = numpy.prod(current_image_dimensions)

            if isinstance(current_layer, MaxPooling):
                current_image_dimensions[0] /= 2
                current_image_dimensions[1] /= 2
                flatten_dimension = current_image_dimensions[0] * current_image_dimensions[1] * current_image_dimensions[2]

            if isinstance(current_layer, FullyConnected):
                samples_per_sample += flatten_dimension * current_layer.output_dim + current_layer.output_dim
                flatten_dimension = current_layer.output_dim

        return samples_per_sample


    def save(self, saver, path):

        session = Session.instance().session

        open(path+"_sequence_net.pickle", 'wb').write(self.pickle_architecture)
        return saver.save(session, path)

    def load(self, path, clear_devices=False, session_from_scratch=False):

        session = Session.instance(new=session_from_scratch).session

        self.sequence_net = pickle.loads(open(path+"_sequence_net.pickle", 'rb').read())
        if clear_devices:
            saver = tf.train.import_meta_graph(path + ".meta", clear_devices=clear_devices)
        else:
            saver = tf.train.import_meta_graph(path + ".meta")

        saver.restore(session, path)
        self.inference_graph = tf.get_collection("inference_graph")[0]
        self.inference_placeholder = tf.get_collection("inference_placeholder")[0]

        return saver
