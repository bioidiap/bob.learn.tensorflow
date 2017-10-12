#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.io.base
import six
import bob.io.image
import bob.ip.base
import bob.core
logger = bob.core.log.setup("bob.learn.tensorflow")

import tensorflow as tf

from .Disk import Disk
from .Triplet import Triplet


class TripletDisk(Triplet, Disk):
    """
     This :py:class:`bob.learn.tensorflow.datashuffler.Triplet` datashuffler deal with databases that are stored in the disk.
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
                 input_shape,
                 input_dtype="float32",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 normalizer=None,
                 prefetch=False,
                 prefetch_capacity=50,
                 prefetch_threads=10
                 ):

        if isinstance(data, list):
            data = numpy.array(data)

        if isinstance(labels, list):
            labels = numpy.array(labels)

        super(TripletDisk, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
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

    def _fetch_batch(self):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        triplets = self.get_triplets(self.data, self.labels)

        for i in range(self.data.shape[0]):

            anchor_filename, positive_filename, negative_filename = six.next(triplets)

            anchor = self.load_from_file(str(anchor_filename))
            positive = self.load_from_file(str(positive_filename))
            negative = self.load_from_file(str(negative_filename))

            # Applying the data augmentation
            if self.data_augmentation is not None:
                    d = self.bob2skimage(self.data_augmentation(self.skimage2bob(anchor)))
                    anchor = d

                    d = self.bob2skimage(self.data_augmentation(self.skimage2bob(positive)))
                    positive = d

                    d = self.bob2skimage(self.data_augmentation(self.skimage2bob(negative)))
                    negative = d

            # Scaling
            anchor = self.normalize_sample(anchor).astype(self.input_dtype)
            positive = self.normalize_sample(positive).astype(self.input_dtype)
            negative = self.normalize_sample(negative).astype(self.input_dtype)

            yield anchor, positive, negative
