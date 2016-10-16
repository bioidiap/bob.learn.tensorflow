#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy

from .Memory import Memory
from .Siamese import Siamese
import tensorflow as tf


class SiameseMemory(Siamese, Memory):

    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float",
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

        super(SiameseMemory, self).__init__(
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

    def get_batch(self, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """
        data = numpy.zeros(shape=self.shape, dtype='float')
        data_p = numpy.zeros(shape=self.shape, dtype='float')
        labels_siamese = numpy.zeros(shape=self.shape[0], dtype='float')

        genuine = True
        for i in range(self.shape[0]):
            data[i, ...], data_p[i, ...] = self.get_genuine_or_not(self.data, self.labels, genuine=genuine)
            if zero_one_labels:
                labels_siamese[i] = not genuine
            else:
                labels_siamese[i] = -1 if genuine else +1
            genuine = not genuine

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(data.shape[0]):
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data[i, ...])))
                data[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_p[i, ...])))
                data_p[i, ...] = d

        # Scaling
        if self.scale:
            data *= self.scale_value
            data_p *= self.scale_value

        return [data.astype("float32"), data_p.astype("float32"), labels_siamese]
