#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
from .Base import Base
from bob.learn.tensorflow.datashuffler.Normalizer import Linear
import tensorflow as tf


class Memory(Base):

    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):
        """
         This datashuffler deal with databases that are stored in a :py:class`numpy.array`

         **Parameters**
           data:
           labels:
           perc_train:
           scale:
           train_batch_size:
           validation_batch_size:
        """

        super(Memory, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer
        )
        # Seting the seed
        numpy.random.seed(seed)
        self.data = self.data.astype(input_dtype)

    def get_batch(self):

        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = self.data[indexes[0:self.batch_size], :, :, :]
        selected_labels = self.labels[indexes[0:self.batch_size]]

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(selected_data.shape[0]):
                img = self.skimage2bob(selected_data[i, ...])
                img = self.data_augmentation(img)
                selected_data[i, ...] = self.bob2skimage(img)

        selected_data = self.normalize_sample(selected_data)

        return [selected_data.astype("float32"), selected_labels.astype("int64")]
