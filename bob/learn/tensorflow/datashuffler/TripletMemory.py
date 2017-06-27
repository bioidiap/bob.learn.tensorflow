#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

from .Memory import Memory
from .Triplet import Triplet
from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class TripletMemory(Triplet, Memory):
    """
     This :py:class:`bob.learn.tensorflow.datashuffler.Triplet` datashuffler deal with databases that are stored in memory
     The data is loaded on the fly.

     **Parameters**

     data:
       Input data to be trainer

     labels:
       Labels. These labels should be set from 0..1

     input_shape:
       The shape of the inputs

     input_dtype:
       The type of the data,

     batch_size:
       Batch size

     seed:
       The seed of the random number generator

     data_augmentation:
       The algorithm used for data augmentation. Look :py:class:`bob.learn.tensorflow.datashuffler.DataAugmentation`

     normalizer:
       The algorithm used for feature scaling. Look :py:class:`bob.learn.tensorflow.datashuffler.ScaleFactor`, :py:class:`bob.learn.tensorflow.datashuffler.Linear` and :py:class:`bob.learn.tensorflow.datashuffler.MeanOffset`

    """

    def __init__(self, data, labels,
                 input_shape=[None, 28, 28, 1],
                 input_dtype="float32",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):

        super(TripletMemory, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer
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

        shape = [self.batch_size] + list(self.input_shape[1:])

        sample_a = numpy.zeros(shape=shape, dtype=self.input_dtype)
        sample_p = numpy.zeros(shape=shape, dtype=self.input_dtype)
        sample_n = numpy.zeros(shape=shape, dtype=self.input_dtype)

        for i in range(shape[0]):
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

        return [sample_a.astype(self.input_dtype), sample_p.astype(self.input_dtype), sample_n.astype(self.input_dtype)]
