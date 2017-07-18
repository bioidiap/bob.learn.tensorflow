#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
from .Base import Base
from bob.learn.tensorflow.datashuffler.Normalizer import Linear
import tensorflow as tf


class Memory(Base):

    """
    This datashuffler deal with memory databases that are stored in a :py:class`numpy.array`

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
                 input_shape,
                 input_dtype="float32",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear(),
                 prefetch=False,
                 prefetch_capacity=10,
                 prefetch_threads=5
                 ):

        super(Memory, self).__init__(
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

    def _fetch_batch(self):
        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        #import ipdb; ipdb.set_trace();

        for i in range(len(indexes)):

            sample = self.data[indexes[i], ...]
            label = self.labels[indexes[i]]

            if self.data_augmentation is not None:
                sample = self.skimage2bob(sample)
                sample = self.data_augmentation(sample)
                sample = self.bob2skimage(sample)

            if self.normalize_sample is not None:
                sample = self.normalize_sample(sample)

            yield [sample, label]

    def get_batch(self):
        """
        Shuffle the Memory dataset and get a random batch.

        ** Returns **

        data:
          Selected samples

        labels:
          Correspondent labels
        """

        if self.generator is None:
            self.generator = self._fetch_batch()

        holder = []
        try:
            for i in range(self.batch_size):
                data = self.generator.next()
                holder.append(data)
                if len(holder) == self.batch_size:
                    return self._aggregate_batch(holder, False)

        except StopIteration:
            self.generator = None
            self.epoch += 1
            return self._aggregate_batch(holder, False)
