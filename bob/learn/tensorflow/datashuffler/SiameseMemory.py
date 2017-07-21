#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy

from .Memory import Memory
from .Siamese import Siamese
import tensorflow as tf
from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class SiameseMemory(Siamese, Memory):
    """
     This :py:class:`bob.learn.tensorflow.datashuffler.Siamese` datashuffler deal with databases that are in Memory
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
                 batch_size=32,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear(),
                 prefetch=False,
                 prefetch_capacity=50,
                 prefetch_threads=10
                 ):

        super(SiameseMemory, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer,
            prefetch=prefetch,
            prefetch_capacity=prefetch_capacity,
            prefetch_threads=prefetch_threads
        )
        # Seting the seed
        numpy.random.seed(seed)
        self.data = self.data.astype(input_dtype)

    def _fetch_batch(self, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        pairs_generator = self.get_genuine_or_not(self.data, self.labels)
        for i in range(self.data.shape[0]):

            left, right, label = pairs_generator.next()

            # Applying the data augmentation
            if self.data_augmentation is not None:
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(left)))
                left = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(right)))
                right = d

            left = self.normalize_sample(left)
            right = self.normalize_sample(right)

            yield left.astype(self.input_dtype), right.astype(self.input_dtype), label
