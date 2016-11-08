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
from bob.learn.tensorflow.datashuffler.Normalizer import Linear

class TripletDisk(Triplet, Disk):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):
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
            batch_size=batch_size,
            data_augmentation=data_augmentation,
            normalizer=normalizer
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

        sample_a = numpy.zeros(shape=self.shape, dtype='float32')
        sample_p = numpy.zeros(shape=self.shape, dtype='float32')
        sample_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            file_name_a, file_name_p, file_name_n = self.get_one_triplet(self.data, self.labels)
            sample_a[i, ...] = self.load_from_file(str(file_name_a))
            sample_p[i, ...] = self.load_from_file(str(file_name_p))
            sample_n[i, ...] = self.load_from_file(str(file_name_n))

        sample_a = self.normalize_sample(sample_a)
        sample_p = self.normalize_sample(sample_p)
        sample_n = self.normalize_sample(sample_n)

        return [sample_a, sample_p, sample_n]
