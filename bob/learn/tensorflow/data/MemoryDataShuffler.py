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

    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 perc_train=0.9,
                 scale=True,
                 train_batch_size=1,
                 validation_batch_size=300):
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
            input_dtype=input_dtype,
            perc_train=perc_train,
            scale=scale,
            train_batch_size=train_batch_size,
            validation_batch_size=validation_batch_size
        )

        self.train_data = self.train_data.astype(input_dtype)
        self.validation_data = self.validation_data.astype(input_dtype)

        if self.scale:
            self.train_data *= self.scale_value
            self.validation_data *= self.scale_value

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

    def get_pair(self, train_dataset=True, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        if train_dataset:
            target_data = self.train_data
            target_labels = self.train_labels
            shape = self.train_shape
        else:
            target_data = self.validation_data
            target_labels = self.validation_labels
            shape = self.validation_shape

        data = numpy.zeros(shape=shape, dtype='float32')
        data_p = numpy.zeros(shape=shape, dtype='float32')
        labels_siamese = numpy.zeros(shape=shape[0], dtype='float32')

        genuine = True
        for i in range(shape[0]):
            data[i, ...], data_p[i, ...] = self.get_genuine_or_not(target_data, target_labels, genuine=genuine)
            if zero_one_labels:
                labels_siamese[i] = not genuine
            else:
                labels_siamese[i] = -1 if genuine else +1
            genuine = not genuine

        return data, data_p, labels_siamese
