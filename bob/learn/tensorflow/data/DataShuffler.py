#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

def scale_mean_norm(data, scale=0.00390625):
    mean = numpy.mean(data)
    data = (data - mean) * scale

    return data, mean


class DataShuffler(object):
    def __init__(self, data, labels, perc_train=0.9, scale=True, train_batch_size=1, validation_batch_size=300):
        """
         The class provide some functionalities for shuffling data

         **Parameters**
           data:
        """

        self.perc_train = perc_train
        self.scale = True
        self.scale_value = 0.00390625
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.data = data
        self.labels = labels # From O to N-1
        self.total_labels = max(labels) + 1

        self.n_samples = self.data.shape[0]
        self.width = self.data.shape[1]
        self.height = self.data.shape[2]
        self.channels = self.data.shape[3]
        self.start_shuffler()

    def get_placeholders(self, name="", train_dataset=True):
        """
        """

        batch = self.train_batch_size if train_dataset else self.validation_batch_size

        data = tf.placeholder(tf.float32, shape=(batch, self.width,
                                                 self.height, self.channels), name=name)

        labels = tf.placeholder(tf.int64, shape=batch)

        return data, labels

    def start_shuffler(self):
        """
         Some base functions for neural networks

         **Parameters**
           data:
        """

        indexes = numpy.array(range(self.n_samples))
        numpy.random.shuffle(indexes)

        # Spliting train and validation
        train_samples = int(round(self.n_samples * self.perc_train))
        validation_samples = self.n_samples - train_samples

        self.train_data = self.data[indexes[0:train_samples], :, :, :]
        self.train_labels = self.labels[indexes[0:train_samples]]

        self.validation_data = self.data[indexes[train_samples:train_samples + validation_samples], :, :, :]
        self.validation_labels = self.labels[indexes[train_samples:train_samples + validation_samples]]

        if self.scale:
            # data = scale_minmax_norm(data,lower_bound = -1, upper_bound = 1)
            self.train_data, self.mean = scale_mean_norm(self.train_data)
            self.validation_data = (self.validation_data - self.mean) * self.scale_value

    def get_batch(self, train_dataset=True):

        if train_dataset:
            n_samples = self.train_batch_size
        else:
            n_samples = self.validation_batch_size

        if train_dataset:
            data = self.train_data
            label = self.train_labels
        else:
            data = self.validation_data
            label = self.validation_labels

        # Shuffling samples
        indexes = numpy.array(range(data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = data[indexes[0:n_samples], :, :, :]
        selected_labels = label[indexes[0:n_samples]]

        return selected_data.astype("float32"), selected_labels
