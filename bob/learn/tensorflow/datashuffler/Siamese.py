#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
from .Base import Base
import tensorflow as tf


class Siamese(Base):
    """
     This datashuffler deal with databases that are provides data to Siamese networks.
     Basically the py:meth:`get_batch` method provides you 3 elements in the returned list.

     The first two are the batch data, and the last is the label. Either `0` for samples from the same class or `1`
     for samples from different class.
      
    
     Here, an epoch is not all possible pairs. An epoch is when you pass thought all the samples at least once.

    """

    def __init__(self, **kwargs):
        super(Siamese, self).__init__(**kwargs)

    def create_placeholders(self):
        """
        Create place holder instances

        :return: 
        """
        with tf.name_scope("Input"):
            self.data_ph = dict()
            self.data_ph['left'] = tf.placeholder(tf.float32, shape=self.input_shape, name="left")
            self.data_ph['right'] = tf.placeholder(tf.float32, shape=self.input_shape, name="right")
            self.label_ph = tf.placeholder(tf.int64, shape=[None], name="label")

            if self.prefetch:
                queue = tf.FIFOQueue(capacity=self.prefetch_capacity,
                                     dtypes=[tf.float32, tf.float32, tf.int64],
                                     shapes=[self.input_shape[1:], self.input_shape[1:], []])

                self.data_ph_from_queue = dict()
                self.data_ph_from_queue['left'] = None
                self.data_ph_from_queue['right'] = None

                # Fetching the place holders from the queue
                self.enqueue_op = queue.enqueue_many([self.data_ph['left'], self.data_ph['right'], self.label_ph])
                self.data_ph_from_queue['left'], self.data_ph_from_queue['right'], self.label_ph_from_queue = queue.dequeue_many(self.batch_size)

            else:
                self.data_ph_from_queue = dict()
                self.data_ph_from_queue['left'] = self.data_ph['left']
                self.data_ph_from_queue['right'] = self.data_ph['right']
                self.label_ph_from_queue = self.label_ph

    def get_genuine_or_not(self, input_data, input_labels):
        """
        Creates a generator with pairs of genuines and and impostors pairs         
        """

        # Shuffling all the indexes
        indexes_per_labels = dict()
        for l in self.possible_labels:
            indexes_per_labels[l] = numpy.where(input_labels == l)[0]
            numpy.random.shuffle(indexes_per_labels[l])

        genuine = True
        for i in range(input_data.shape[0]):

            if genuine:
                # Selecting the class
                class_index = numpy.random.randint(len(self.possible_labels))

                # Now selecting the samples for the pair
                left = input_data[indexes_per_labels[class_index][numpy.random.randint(len(indexes_per_labels[class_index]))]]
                right = input_data[indexes_per_labels[class_index][numpy.random.randint(len(indexes_per_labels[class_index]))]]

                yield left, right, 0

            else:
                # Selecting the 2 classes
                class_index = numpy.random.choice(len(self.possible_labels), 2, replace=False)

                # Now selecting the samples for the pair
                left = input_data[indexes_per_labels[class_index[0]][numpy.random.randint(len(indexes_per_labels[class_index[0]]))]]
                right = input_data[indexes_per_labels[class_index[1]][numpy.random.randint(len(indexes_per_labels[class_index[1]]))]]

                yield left, right, 1

            genuine = not genuine
