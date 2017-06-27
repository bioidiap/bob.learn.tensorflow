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
                 normalizer=Linear()):

        super(Memory, self).__init__(
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
        Shuffle the Memory dataset and get a random batch.

        ** Returns **

        data:
          Selected samples

        labels:
          Correspondent labels
        """
        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = self.data[indexes[0:self.batch_size], ...]
        selected_labels = self.labels[indexes[0:self.batch_size]]

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(selected_data.shape[0]):
                img = self.skimage2bob(selected_data[i, ...])
                img = self.data_augmentation(img)
                selected_data[i, ...] = self.bob2skimage(img)

        selected_data = self.normalize_sample(selected_data)

        return [selected_data.astype("float32"), selected_labels.astype("int64")]
