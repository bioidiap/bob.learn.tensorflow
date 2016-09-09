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
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1):
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

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(TextDataShuffler, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size
        )

    def load_from_file(self, file_name, shape):
        d = bob.io.base.load(file_name)
        if len(d.shape) == 2:
            data = numpy.zeros(shape=tuple(shape[1:]))
            data[:, :, 0] = d
        else:
            data = d

        return data

    def get_batch(self):

        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = numpy.zeros(shape=self.shape)
        for i in range(self.batch_size):

            file_name = self.data[indexes[i]]
            data = self.load_from_file(file_name, self.shape)

            selected_data[i, ...] = data
            if self.scale:
                selected_data[i, ...] *= self.scale_value

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
            file_name, file_name_p = self.get_genuine_or_not(self.data, self.labels, genuine=genuine)
            data[i, ...] = self.load_from_file(str(file_name), self.shape)
            data_p[i, ...] = self.load_from_file(str(file_name_p), self.shape)

            if zero_one_labels:
                labels_siamese[i] = not genuine
            else:
                labels_siamese[i] = -1 if genuine else +1
            genuine = not genuine

        if self.scale:
            data *= self.scale_value
            data_p *= self.scale_value

        return data, data_p, labels_siamese

