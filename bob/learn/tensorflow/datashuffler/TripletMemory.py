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

        sample_a = numpy.zeros(shape=self.shape, dtype='float32')
        sample_p = numpy.zeros(shape=self.shape, dtype='float32')
        sample_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            sample_a[i, ...], sample_p[i, ...], sample_n[i, ...] = self.get_one_triplet(self.data, self.labels)

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(sample_a.shape[0]):
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(sample_a[i, ...])))
                sample_a[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(sample_p[i, ...])))
                sample_p[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(sample_n[i, ...])))
                sample_n[i, ...] = d

        # Scaling
        sample_a = self.normalize_sample(sample_a)
        sample_p = self.normalize_sample(sample_p)
        sample_n = self.normalize_sample(sample_n)

        return [sample_a.astype("float32"), sample_p.astype("float32"), sample_n.astype("float32")]
