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
                 input_dtype="float32",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=Linear(),
                 prefetch=False,
                 prefetch_capacity=10,
                 prefetch_threads=5
                 ):

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
            normalizer=normalizer,
            prefetch=prefetch,
            prefetch_capacity=prefetch_capacity,
            prefetch_threads=prefetch_threads
        )
        # Seting the seed
        numpy.random.seed(seed)

        # TODO: very bad solution to deal with bob.shape images an tf shape images
        self.bob_shape = tuple([input_shape[3]] + list(input_shape[1:3]))

    def _fetch_batch(self, zero_one_labels=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        pairs_generator = self.get_genuine_or_not(self.data, self.labels)
        for i in range(self.data.shape[0]):

            left_filename, right_filename, label = pairs_generator.next()
            left = self.load_from_file(left_filename)
            right = self.load_from_file(right_filename)

            # Applying the data augmentation
            if self.data_augmentation is not None:
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(left)))
                left = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(right)))
                right = d

            left = self.normalize_sample(left)
            right = self.normalize_sample(right)

            yield left.astype(self.input_dtype), right.astype(self.input_dtype), label
