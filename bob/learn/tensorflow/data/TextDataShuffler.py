#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.io.base
import bob.io.image
import tensorflow as tf

from .BaseDataShuffler import BaseDataShuffler

#def scale_mean_norm(data, scale=0.00390625):
#    mean = numpy.mean(data)
#    data = (data - mean) * scale

#    return data, mean


class TextDataShuffler(BaseDataShuffler):
    def __init__(self, data, labels, input_shape, perc_train=0.9, scale=0.00390625, train_batch_size=1, validation_batch_size=300):
        """
         Shuffler that deal with file list

         **Parameters**
           data:
           labels:
           perc_train:
           scale:
           train_batch_size:
           validation_batch_size:
        """

        super(TextDataShuffler, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            perc_train=perc_train,
            scale=scale,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size
        )

        if isinstance(self.data, list):
            self.data = numpy.array(self.data)

        if isinstance(self.labels, list):
            self.labels = numpy.array(self.labels)

        # Spliting between train and test
        self.train_data = self.data[self.indexes[0:self.n_train_samples]]
        self.train_labels = self.labels[self.indexes[0:self.n_train_samples]]

        self.validation_data = self.data[self.indexes[self.n_train_samples:
                                         self.n_train_samples + self.n_validation_samples]]
        self.validation_labels = self.labels[self.indexes[self.n_train_samples:
                                             self.n_train_samples + self.n_validation_samples]]

    def get_batch(self, train_dataset=True):

        if train_dataset:
            batch_size = self.train_batch_size
            shape = self.train_shape
            files_names = self.train_data
            label = self.train_labels
        else:
            batch_size = self.validation_batch_size
            shape = self.validation_shape
            files_names = self.validation_data
            label = self.validation_labels

        # Shuffling samples
        indexes = numpy.array(range(files_names.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = numpy.zeros(shape=shape)
        for i in range(batch_size):

            file_name = files_names[indexes[i]]

            d = bob.io.base.load(file_name)
            if len(d.shape) == 2:
                data = numpy.zeros(shape=tuple(shape[1:]))
                data[:, :, 0] = d
            else:
                data = d

            selected_data[i, ...] = data
            if self.scale is not None:
                selected_data[i, ...] *= self.scale



        selected_labels = label[indexes[0:batch_size]]

        return selected_data.astype("float32"), selected_labels
