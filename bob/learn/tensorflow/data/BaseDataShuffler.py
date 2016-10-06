#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf


class BaseDataShuffler(object):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1):
        """
         The class provide base functionoalies to shuffle the data

         **Parameters**
           data:
           labels:
           perc_train:
           scale:
           train_batch_size:
           validation_batch_size:
        """

        self.scale = scale
        self.scale_value = 0.00390625
        self.input_dtype = input_dtype

        # TODO: Check if the bacth size is higher than the input data
        self.batch_size = batch_size

        self.data = data
        self.shape = tuple([batch_size] + input_shape)
        self.input_shape = tuple(input_shape)


        self.labels = labels
        self.possible_labels = list(set(self.labels))

        # Computing the data samples fro train and validation
        self.n_samples = len(self.labels)

        # Shuffling all the indexes
        self.indexes = numpy.array(range(self.n_samples))
        numpy.random.shuffle(self.indexes)

    def get_placeholders_forprefetch(self, name=""):
        """
        Returns a place holder with the size of your batch
        """
        data = tf.placeholder(tf.float32, shape=tuple([None] + list(self.shape[1:])), name=name)
        labels = tf.placeholder(tf.int64, shape=[None, ])

        return data, labels

    def get_placeholders(self, name=""):
        """
        Returns a place holder with the size of your batch
        """
        data = tf.placeholder(tf.float32, shape=self.shape, name=name)
        labels = tf.placeholder(tf.int64, shape=self.shape[0])

        return data, labels

    def get_genuine_or_not(self, input_data, input_labels, genuine=True):

        if genuine:
            # Getting a client
            index = numpy.random.randint(len(self.possible_labels))
            index = int(self.possible_labels[index])

            # Getting the indexes of the data from a particular client
            indexes = numpy.where(input_labels == index)[0]
            numpy.random.shuffle(indexes)

            # Picking a pair
            data = input_data[indexes[0], ...]
            data_p = input_data[indexes[1], ...]

        else:
            # Picking a pair of labels from different clients
            index = numpy.random.choice(len(self.possible_labels), 2, replace=False)
            index[0] = self.possible_labels[int(index[0])]
            index[1] = self.possible_labels[int(index[1])]

            # Getting the indexes of the two clients
            indexes = numpy.where(input_labels == index[0])[0]
            indexes_p = numpy.where(input_labels == index[1])[0]
            numpy.random.shuffle(indexes)
            numpy.random.shuffle(indexes_p)

            # Picking a pair
            data = input_data[indexes[0], ...]
            data_p = input_data[indexes_p[0], ...]

        return data, data_p

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

