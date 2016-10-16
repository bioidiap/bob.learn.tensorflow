#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

from .Memory import Memory
from Triplet import Triplet


class TripletMemory(Triplet, Memory):

    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10,
                 data_augmentation=None):
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

        super(TripletMemory, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation
        )
        # Seting the seed
        numpy.random.seed(seed)

        self.data = self.data.astype(input_dtype)

    def get_batch(self):
        """
        Get a random triplet

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            data_a[i, ...], data_p[i, ...], data_n[i, ...] = self.get_one_triplet(self.data, self.labels)

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(data_a.shape[0]):
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_a[i, ...])))
                data_a[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_p[i, ...])))
                data_p[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_n[i, ...])))
                data_n[i, ...] = d

        # Scaling
        if self.scale:
            data_a *= self.scale_value
            data_p *= self.scale_value
            data_n *= self.scale_value

        return [data_a.astype("float32"), data_p.astype("float32"), data_n.astype("float32")]
