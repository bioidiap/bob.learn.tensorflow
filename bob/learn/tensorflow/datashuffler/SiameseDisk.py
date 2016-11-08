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
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear()):
        """
         Shuffler that deal with file list

         **Parameters**
         data:
         labels:
         input_shape: Shape of the input. `input_shape != data.shape`, the data will be reshaped
         input_dtype="float64":
         scale=True:
         batch_size=1:
        """

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(SiameseDisk, self).__init__(
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
            sample_l[i, ...] = self.load_from_file(str(file_name))
            sample_r[i, ...] = self.load_from_file(str(file_name_p))

            labels_siamese[i] = not genuine
            genuine = not genuine

        sample_l = self.normalize_sample(sample_l)
        sample_r = self.normalize_sample(sample_r)

        return sample_l, sample_r, labels_siamese
