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
       data: Input data to be trainer
       labels: Labels. These labels should be set from 0..1
       input_shape: The shape of the inputs
       input_dtype: The type of the data,
       batch_size: Batch size
       seed: The seed of the random number generator
       data_augmentation: The algorithm used for data augmentation. Look :py:class:`bob.learn.tensorflow.datashuffler.DataAugmentation`
       normalizer: The algorithm used for feature scaling. Look :py:class:`bob.learn.tensorflow.datashuffler.ScaleFactor`, :py:class:`bob.learn.tensorflow.datashuffler.Linear` and :py:class:`bob.learn.tensorflow.datashuffler.MeanOffset`
    """

    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):

        super(SiameseMemory, self).__init__(
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

    def get_batch(self, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """
        sample_l = numpy.zeros(shape=self.shape, dtype='float')
        sample_r = numpy.zeros(shape=self.shape, dtype='float')
        labels_siamese = numpy.zeros(shape=self.shape[0], dtype='float')

        genuine = True
        for i in range(self.shape[0]):
            sample_l[i, ...], sample_r[i, ...] = self.get_genuine_or_not(self.data, self.labels, genuine=genuine)
            if zero_one_labels:
                labels_siamese[i] = not genuine
            else:
                labels_siamese[i] = -1 if genuine else +1
            genuine = not genuine

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(sample_l.shape[0]):
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(sample_l[i, ...])))
                sample_l[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(sample_r[i, ...])))
                sample_r[i, ...] = d

        sample_l = self.normalize_sample(sample_l)
        sample_r = self.normalize_sample(sample_r)

        return [sample_l.astype("float32"), sample_r.astype("float32"), labels_siamese]
