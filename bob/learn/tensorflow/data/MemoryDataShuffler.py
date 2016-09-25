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
                 scale=True,
                 batch_size=1):
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
            scale=scale,
            batch_size=batch_size
        )

        self.data = self.data.astype(input_dtype)
        if self.scale:
            self.data *= self.scale_value

    def get_batch(self):

        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = self.data[indexes[0:self.batch_size], :, :, :]
        selected_labels = self.labels[indexes[0:self.batch_size]]

        return selected_data.astype("float32"), selected_labels

    def get_pair(self, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """
        data = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        labels_siamese = numpy.zeros(shape=self.shape[0], dtype='float32')

        genuine = True
        for i in range(self.shape[0]):
            data[i, ...], data_p[i, ...] = self.get_genuine_or_not(self.data, self.labels, genuine=genuine)
            if zero_one_labels:
                labels_siamese[i] = not genuine
            else:
                labels_siamese[i] = -1 if genuine else +1
            genuine = not genuine

        return data, data_p, labels_siamese

    def get_random_triplet(self, n_triplets=1):
        """
        Get a random triplet

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(n_triplets):
            data_a[i, ...], data_p[i, ...], data_n[i, ...] = self.get_one_triplet(self.data, self.labels)

        return data_a, data_p, data_n

