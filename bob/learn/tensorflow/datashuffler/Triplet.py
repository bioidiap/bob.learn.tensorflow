#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
from .Base import Base
import tensorflow as tf


class Triplet(Base):
    """
     This datashuffler deal with databases that are provides data to Triplet networks.
     Basically the py:meth:`get_batch` method provides you 3 elements in the returned list.

     The first element is the batch for the anchor, the second one is the batch for the positive class, w.r.t the
     anchor, and  the last one is the batch for the negative class , w.r.t the anchor.

    """

    def __init__(self, **kwargs):
        super(Triplet, self).__init__(**kwargs)
        self.data2_placeholder = None
        self.data3_placeholder = None

    def get_placeholders(self, name=""):
        """
        Returns a place holder with the size of your batch
        """
        if self.data_placeholder is None:
            self.data_placeholder = tf.placeholder(tf.float32, shape=self.shape, name=name+"_anchor")

        if self.data2_placeholder is None:
            self.data2_placeholder = tf.placeholder(tf.float32, shape=self.shape, name=name+"_positive")

        if self.data3_placeholder is None:
            self.data3_placeholder = tf.placeholder(tf.float32, shape=self.shape, name=name+"_negative")

        return [self.data_placeholder, self.data2_placeholder, self.data3_placeholder]

    def get_placeholders_triplet_forprefetch(self, name=""):
        """
        Returns a place holder with the size of your batch
        """
        if self.data_placeholder is None:
            self.data_placeholder = tf.placeholder(tf.float32, shape=tuple([None] + list(self.shape[1:])), name=name)

        if self.data2_placeholder is None:
            self.data2_placeholder = tf.placeholder(tf.float32, shape=tuple([None] + list(self.shape[1:])), name=name)

        if self.data3_placeholder is None:
            self.data3_placeholder = tf.placeholder(tf.float32, shape=tuple([None] + list(self.shape[1:])), name=name)

        return [self.data_placeholder, self.data2_placeholder, self.data3_placeholder]

    def get_one_triplet(self, input_data, input_labels):
        # Getting a pair of clients
        index = numpy.random.choice(len(self.possible_labels), 2, replace=False)
        index[0] = self.possible_labels[index[0]]
        index[1] = self.possible_labels[index[1]]

        # Getting the indexes of the data from a particular client
        indexes = numpy.where(input_labels == index[0])[0]
        numpy.random.shuffle(indexes)

        # Picking a positive pair
        data_anchor = input_data[indexes[0], ...]
        data_positive = input_data[indexes[1], ...]

        # Picking a negative sample
        indexes = numpy.where(input_labels == index[1])[0]
        numpy.random.shuffle(indexes)
        data_negative = input_data[indexes[0], ...]

        return data_anchor, data_positive, data_negative
