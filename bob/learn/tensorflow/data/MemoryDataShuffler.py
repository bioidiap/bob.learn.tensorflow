#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

from .BaseDataShuffler import BaseDataShuffler

def scale_mean_norm(data, scale=0.00390625):
    mean = numpy.mean(data)
    data = (data - mean) * scale

    return data, mean


class MemoryDataShuffler(BaseDataShuffler):
    def __init__(self, data, labels, input_shape, perc_train=0.9, scale=True, train_batch_size=1, validation_batch_size=300):
        """
         Shuffler that deal with memory datasets

         **Parameters**
           data:
           labels:
           perc_train:
           scale:
           train_batch_size:
           validation_batch_size:
        """

        super(MemoryDataShuffler, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            perc_train=perc_train,
            scale=scale,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size
        )

        # Spliting between train and test
        self.train_data = self.data[self.indexes[0:self.n_train_samples], ...]
        self.train_labels = self.labels[self.indexes[0:self.n_train_samples]]

        self.validation_data = self.data[self.indexes[self.n_train_samples:
                                         self.n_train_samples + self.n_validation_samples], ...]
        self.validation_labels = self.labels[self.indexes[self.n_train_samples:
                                             self.n_train_samples + self.n_validation_samples]]
        if self.scale:
            self.train_data, self.mean = scale_mean_norm(self.train_data)
            self.validation_data = (self.validation_data - self.mean) * self.scale_value

    def get_batch(self, train_dataset=True):

        if train_dataset:
            n_samples = self.train_batch_size
            data = self.train_data
            label = self.train_labels
        else:
            n_samples = self.validation_batch_size
            data = self.validation_data
            label = self.validation_labels

        # Shuffling samples
        indexes = numpy.array(range(data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = data[indexes[0:n_samples], :, :, :]
        selected_labels = label[indexes[0:n_samples]]

        return selected_data.astype("float32"), selected_labels
