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
                 perc_train=0.9,
                 scale=True,
                 train_batch_size=1,
                 validation_batch_size=300):
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
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size

        self.data = data
        self.train_shape = tuple([train_batch_size] + input_shape)
        self.validation_shape = tuple([validation_batch_size] + input_shape)

        # TODO: Check if the labels goes from O to N-1
        self.labels = labels
        self.possible_labels = list(set(self.labels))

        # Computing the data samples fro train and validation
        self.n_samples = len(self.labels)
        self.n_train_samples = int(round(self.n_samples * perc_train))
        self.n_validation_samples = self.n_samples - self.n_train_samples

        # Shuffling all the indexes
        self.indexes = numpy.array(range(self.n_samples))
        numpy.random.shuffle(self.indexes)

        # Spliting the data between train and validation
        self.train_data = self.data[self.indexes[0:self.n_train_samples], ...]
        self.train_labels = self.labels[self.indexes[0:self.n_train_samples]]

        self.validation_data = self.data[self.indexes[self.n_train_samples:
                                         self.n_train_samples + self.n_validation_samples], ...]
        self.validation_labels = self.labels[self.indexes[self.n_train_samples:
                                         self.n_train_samples + self.n_validation_samples]]

    def get_placeholders_forprefetch(self, name="", train_dataset=True):
        """
        Returns a place holder with the size of your batch
        """

        shape = self.train_shape if train_dataset else self.validation_shape
        data = tf.placeholder(tf.float32, shape=tuple([None] + list(shape[1:])), name=name)
        labels = tf.placeholder(tf.int64, shape=[None, ])

        return data, labels

    def get_placeholders(self, name="", train_dataset=True):
        """
        Returns a place holder with the size of your batch
        """

        shape = self.train_shape if train_dataset else self.validation_shape
        data = tf.placeholder(tf.float32, shape=shape, name=name)
        labels = tf.placeholder(tf.int64, shape=shape[0])

        return data, labels

    def get_genuine_or_not(self, input_data, input_labels, genuine=True):
        if genuine:
            # Getting a client
            index = numpy.random.randint(len(self.possible_labels))
            index = self.possible_labels[index]

            # Getting the indexes of the data from a particular client
            indexes = numpy.where(input_labels == index)[0]
            numpy.random.shuffle(indexes)

            # Picking a pair
            data = input_data[indexes[0], ...]
            data_p = input_data[indexes[1], ...]

        else:
            # Picking a pair of labels from different clients
            index = numpy.random.choice(len(self.possible_labels), 2, replace=False)
            index[0] = self.possible_labels[index[0]]
            index[1] = self.possible_labels[index[1]]

            # Getting the indexes of the two clients
            indexes = numpy.where(input_labels == index[0])[0]
            indexes_p = numpy.where(input_labels == index[1])[0]
            numpy.random.shuffle(indexes)
            numpy.random.shuffle(indexes_p)

            # Picking a pair
            data = input_data[indexes[0], ...]
            data_p = input_data[indexes_p[0], ...]

        return data, data_p
