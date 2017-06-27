#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
from .Base import Base
import tensorflow as tf


class Siamese(Base):
    """
     This datashuffler deal with databases that are provides data to Siamese networks.
     Basically the py:meth:`get_batch` method provides you 3 elements in the returned list.

     The first two are the batch data, and the last is the label. Either `0` for samples from the same class or `1`
      for samples from different class.

    """

    def __init__(self, **kwargs):
        super(Siamese, self).__init__(**kwargs)

    def create_placeholders(self):
        """
        Create place holder instances

        :return: 
        """
        with tf.name_scope("Input"):
            self.data_ph = dict()
            self.data_ph['left'] = tf.placeholder(tf.float32, shape=self.input_shape, name="left")
            self.data_ph['right'] = tf.placeholder(tf.float32, shape=self.input_shape, name="right")
            self.label_ph = tf.placeholder(tf.int64, shape=[None], name="label")

            # If prefetch, setup the queue to feed data
            if self.prefetch:
                raise ValueError("There is no prefetch for siamease networks")

    def get_genuine_or_not(self, input_data, input_labels, genuine=True):

        if genuine:
            # Getting a client
            index = numpy.random.randint(len(self.possible_labels))
            index = int(self.possible_labels[index])

            # Getting the indexes of the data from a particular client
            indexes = numpy.where(input_labels == index)[0]
            numpy.random.shuffle(indexes)

            # Picking a pair
            sample_l = input_data[indexes[0], ...]
            sample_r = input_data[indexes[1], ...]

        else:
            # Picking a pair of labels from different clients
            index = numpy.random.choice(len(self.possible_labels), 2, replace=False)
            index[0] = self.possible_labels[int(index[0])]
            index[1] = self.possible_labels[int(index[1])]

            # Getting the indexes of the two clients
            indexes_l = numpy.where(input_labels == index[0])[0]
            indexes_r = numpy.where(input_labels == index[1])[0]
            numpy.random.shuffle(indexes_l)
            numpy.random.shuffle(indexes_r)

            # Picking a pair
            sample_l = input_data[indexes_l[0], ...]
            sample_r = input_data[indexes_r[0], ...]

        return sample_l, sample_r
