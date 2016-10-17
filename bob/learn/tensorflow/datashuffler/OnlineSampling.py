#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
from bob.learn.tensorflow.network import SequenceNetwork


class OnLineSampling(object):
    """
    This data shuffler uses the current state of the network to select the samples.
    This class is not meant to be used, but extended.

    For instance, this is used for triplet selection :py:class:`bob.learn.tensorflow.datashuffler.Triplet`
    """

    def __init__(self, **kwargs):
        self.feature_extractor = None
        self.session = None
        self.feature_placeholder = None
        self.graph = None

    def clear_variables(self):
        self.feature_extractor = None
        self.session = None
        self.feature_placeholder = None
        self.graph = None

    def set_feature_extractor(self, feature_extractor, session=None):
        """
        Set the current feature extraction used in the sampling
        """
        if not isinstance(feature_extractor, SequenceNetwork):
            raise ValueError("Feature extractor must be a `bob.learn.tensoflow.network.SequenceNetwork` object")

        self.feature_extractor = feature_extractor
        self.session = session

    def project(self, data):
        # Feeding the placeholder

        if self.feature_placeholder is None:
            self.feature_placeholder = tf.placeholder(tf.float32, shape=data.shape, name="feature")
            self.graph = self.feature_extractor.compute_graph(self.feature_placeholder, self.feature_extractor.default_feature_layer,
                                                              training=False)

        feed_dict = {self.feature_placeholder: data}
        return self.session.run([self.graph], feed_dict=feed_dict)[0]