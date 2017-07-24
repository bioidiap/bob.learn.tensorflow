#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
from .Base import Base
from bob.learn.tensorflow.datashuffler.Normalizer import Linear
import tensorflow as tf


class Memory(Base):

    """
    This datashuffler deal with memory databases that are stored in a :py:class`numpy.array`

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

        super(Memory, self).__init__(
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
        
        # number of training examples as a 'list'
        self.indexes = numpy.array(range(self.data.shape[0]))
        # shuffle the indexes to get randomized mini-batches
        numpy.random.shuffle(self.indexes)

    def get_batch(self):
        """
        Shuffle the Memory dataset and get a random batch.

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

            selected_data[i, ...] = self.data[indexes[i], ...]
            # Scaling
            selected_data[i, ...] = self.normalize_sample(selected_data[i, ...])

        selected_labels = self.labels[indexes[0:self.batch_size]]

        

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(selected_data.shape[0]):
                img = self.skimage2bob(selected_data[i, ...])
                img = self.data_augmentation(img)
                selected_data[i, ...] = self.bob2skimage(img)

        selected_data = self.normalize_sample(selected_data)

        return [selected_data.astype("float32"), selected_labels.astype("int64")]

  
    def get_batch_epoch(self):
      """get_batch_epoch() -> selected_data, selected_labels

      This function selects and returns data to be used in a minibatch iterations.
      Note that it works in epochs, i.e. all the training data should be seen
      during one epoch, which consists in several minibatch iterations.

      **Returns**

      selected_data:
        Selected samples

      selected_labels:
        Correspondent labels
      """
      # this is done to rebuild the whole list (i.e. at the end of one epoch)
      epoch_done = False

      # returned mini-batch
      selected_data = numpy.zeros(shape=self.shape)
      selected_labels = [] 

      # if there is not enough available data to fill the current mini-batch
      # add randomly some examples THAT ARE NOT STILL PRESENT in the dataset !
      if len(self.indexes) < self.batch_size:

        print "should add examples to the current minibatch {0}".format(len(self.indexes))
        # since we reached the end of an epoch, we'll have to reconsider all the data
        epoch_done = True
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
        selected_data[i, ...] = self.data[self.indexes[current_index], ...]
        
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
      if epoch_done:
        self.indexes = numpy.array(range(self.data.shape[0]))
        numpy.random.shuffle(self.indexes)

      return [selected_data.astype("float32"), selected_labels.astype("int64"), epoch_done]
