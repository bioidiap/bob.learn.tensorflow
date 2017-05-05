#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import bob.io.base
import bob.io.image
import bob.ip.base
import bob.core
from .Base import Base

logger = bob.core.log.setup("bob.learn.tensorflow")
from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class Disk(Base):

    """
     This datashuffler deal with databases that are stored in the disk.
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

        super(Disk, self).__init__(
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

    def load_from_file(self, file_name):
        d = bob.io.base.load(file_name)

        # Applying the data augmentation
        if self.data_augmentation is not None:
            d = self.data_augmentation(d)

        if d.shape[0] != 3 and self.input_shape[2] != 3: # GRAY SCALE IMAGE
            data = numpy.zeros(shape=(d.shape[0], d.shape[1], 1))
            data[:, :, 0] = d
            data = self.rescale(data)
        else:
            d = self.rescale(d)
            data = self.bob2skimage(d)

        # Checking NaN
        if numpy.sum(numpy.isnan(data)) > 0:
            logger.warning("######### Sample {0} has noise #########".format(file_name))

        return data

    def get_batch(self):
        """
        Shuffle the Disk dataset, get a random batch and load it on the fly.

        ** Returns **

        data:
          Selected samples

        labels:
          Correspondent labels
        """

        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = numpy.zeros(shape=self.shape)
        for i in range(self.batch_size):

            file_name = self.data[indexes[i]]
            data = self.load_from_file(file_name)

            selected_data[i, ...] = data

            # Scaling
            selected_data[i, ...] = self.normalize_sample(selected_data[i, ...])

        selected_labels = self.labels[indexes[0:self.batch_size]]

        return [selected_data.astype("float32"), selected_labels.astype("int64")]
