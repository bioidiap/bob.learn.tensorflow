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


class Disk(Base):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10):
        """
         This datashuffler deal with databases that are stored in the disk.
         The data is loaded on the fly,.

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

        super(Disk, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size,
            seed=seed
        )
        # Seting the seed
        numpy.random.seed(seed)

        # TODO: very bad solution to deal with bob.shape images an tf shape images
        self.bob_shape = tuple([input_shape[2]] + list(input_shape[0:2]))

    def load_from_file(self, file_name):
        d = bob.io.base.load(file_name)
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

        # Shuffling samples
        indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = numpy.zeros(shape=self.shape)
        for i in range(self.batch_size):

            file_name = self.data[indexes[i]]
            data = self.load_from_file(file_name)

            selected_data[i, ...] = data
            if self.scale:
                selected_data[i, ...] *= self.scale_value

        selected_labels = self.labels[indexes[0:self.batch_size]]

        return [selected_data.astype("float32"), selected_labels.astype("int64")]
