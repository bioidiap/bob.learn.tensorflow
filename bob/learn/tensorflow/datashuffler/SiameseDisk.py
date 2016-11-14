#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.core
logger = bob.core.log.setup("bob.learn.tensorflow")

from .Disk import Disk
from .Siamese import Siamese

from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class SiameseDisk(Siamese, Disk):
    """
     This :py:class:`bob.learn.tensorflow.datashuffler.Siamese` datashuffler deal with databases that are stored in the disk.
     The data is loaded on the fly,.

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
                 input_dtype="float64",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(SiameseDisk, self).__init__(
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

        # TODO: very bad solution to deal with bob.shape images an tf shape images
        self.bob_shape = tuple([input_shape[2]] + list(input_shape[0:2]))

    def get_batch(self):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        sample_l = numpy.zeros(shape=self.shape, dtype='float32')
        sample_r = numpy.zeros(shape=self.shape, dtype='float32')
        labels_siamese = numpy.zeros(shape=self.shape[0], dtype='float32')

        genuine = True
        for i in range(self.shape[0]):
            file_name, file_name_p = self.get_genuine_or_not(self.data, self.labels, genuine=genuine)
            sample_l[i, ...] = self.normalize_sample(self.load_from_file(str(file_name)))
            sample_r[i, ...] = self.normalize_sample(self.load_from_file(str(file_name_p)))

            labels_siamese[i] = not genuine
            genuine = not genuine

        return sample_l, sample_r, labels_siamese
