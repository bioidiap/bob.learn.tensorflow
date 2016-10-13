#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.io.base
import bob.io.image
import bob.ip.base
import bob.core
logger = bob.core.log.setup("bob.learn.tensorflow")

import tensorflow as tf

from .Disk import Disk
from .Triplet import Triplet


class TripletDisk(Triplet, Disk):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10):
        """
         Shuffler that deal with file list

         **Parameters**
         data:
         labels:
         input_shape: Shape of the input. `input_shape != data.shape`, the data will be reshaped
         input_dtype="float64":
         scale=True:
         batch_size=1:
        """

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(TripletDisk, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size
        )
        # Seting the seed
        numpy.random.seed(seed)

        # TODO: very bad solution to deal with bob.shape images an tf shape images
        self.bob_shape = tuple([input_shape[2]] + list(input_shape[0:2]))

    def get_batch(self):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            file_name_a, file_name_p, file_name_n = self.get_one_triplet(self.data, self.labels)
            data_a[i, ...] = self.load_from_file(str(file_name_a), self.shape)
            data_p[i, ...] = self.load_from_file(str(file_name_p), self.shape)
            data_n[i, ...] = self.load_from_file(str(file_name_n), self.shape)

        if self.scale:
            data_a *= self.scale_value
            data_p *= self.scale_value
            data_n *= self.scale_value

        return [data_a, data_p, data_n]
