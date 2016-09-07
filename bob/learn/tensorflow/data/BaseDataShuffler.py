#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf


class BaseDataShuffler(object):
    def __init__(self, data, labels, input_shape, perc_train=0.9, scale=True, train_batch_size=1, validation_batch_size=300):
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

        # TODO: Check if the bacth size is higher than the input data
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size

        self.data = data
        self.train_shape = tuple([train_batch_size] + input_shape)
        self.validation_shape = tuple([validation_batch_size] + input_shape)

        # TODO: Check if the labels goes from O to N-1
        self.labels = labels
        self.total_labels = max(labels) + 1

        # Spliting in train and validation
        self.n_samples = len(self.labels)
        self.n_train_samples = int(round(self.n_samples * perc_train))
        self.n_validation_samples = self.n_samples - self.n_train_samples

        # Shuffling all the indexes
        self.indexes = numpy.array(range(self.n_samples))
        numpy.random.shuffle(self.indexes)

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
