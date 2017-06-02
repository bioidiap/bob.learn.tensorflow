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

        # number of training examples as a 'list'
        self.indexes = numpy.array(range(self.data.shape[0]))
        # shuffle the indexes to get randomized mini-batches
        numpy.random.shuffle(self.indexes)

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


    def get_batch_epoch(self):

      # this is done to rebuild the whole list (i.e. at the end of one epoch)
      rebuild_indexes = False

      # returned mini-batch
      selected_data = numpy.zeros(shape=self.shape)
      selected_labels = [] 

      # if there is not enough available data to fill the current mini-batch
      # add randomly some examples THAT ARE NOT STILL PRESENT in the dataset !
      if len(self.indexes) < self.batch_size:

        print "should add examples to the current minibatch {0}".format(len(self.indexes))
        # since we reached the end of an epoch, we'll hace to reconsider all the data
        rebuild_indexes = True
        number_of_examples_to_add = self.batch_size - len(self.indexes) 
        added_examples = 0
        
        # generate a list of potential examples to add to this mini-batch
        potential_indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(potential_indexes)
        
        # add indexes that are not still present in the training data
        for pot_index in potential_indexes:
          if pot_index not in self.indexes:
            self.indexes = numpy.append(self.indexes, [pot_index])
            added_examples += 1
            
            # stop if we have enough examples
            if added_examples == number_of_examples_to_add:
              break
      
      # populate mini-batch
      for i in range(self.batch_size):

        current_index = self.batch_size - i - 1
        
        # get the data example
        file_name = self.data[self.indexes[current_index]]
        data = self.load_from_file(file_name)
        selected_data[i, ...] = data
        
        # normalization
        selected_data[i, ...] = self.normalize_sample(selected_data[i, ...])
        
        # label
        selected_labels.append(self.labels[self.indexes[current_index]])

        # remove this example from the training set - used once in the epoch
        new_indexes = numpy.delete(self.indexes, current_index)
        self.indexes = new_indexes

      if isinstance(selected_labels, list):
        selected_labels = numpy.array(selected_labels)

      # rebuild whole randomly shuffled training dataset
      if rebuild_indexes:
        self.indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(self.indexes)

      return [selected_data.astype("float32"), selected_labels.astype("int64")]
