#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST

import numpy
from .Base import Base
import tensorflow as tf


class Triplet(Base):
    """
     This datashuffler deal with databases that are provides data to Triplet networks.
     Basically the py:meth:`get_batch` method provides you 3 elements in the returned list.

     The first element is the batch for the anchor, the second one is the batch for the positive class, w.r.t the
     anchor, and  the last one is the batch for the negative class , w.r.t the anchor.
     
     Here, an epoch is not all possible triplets. An epoch is when you pass thought all the samples at least once.

    """

    def __init__(self, **kwargs):
        super(Triplet, self).__init__(**kwargs)

    def create_placeholders(self):
        """
        Create place holder instances

        :return: 
        """
        with tf.name_scope("Input"):
            self.data_ph = dict()
            self.data_ph['anchor'] = tf.placeholder(
                tf.float32, shape=self.input_shape, name="anchor")
            self.data_ph['positive'] = tf.placeholder(
                tf.float32, shape=self.input_shape, name="positive")
            self.data_ph['negative'] = tf.placeholder(
                tf.float32, shape=self.input_shape, name="negative")

            if self.prefetch:
                queue = tf.FIFOQueue(
                    capacity=self.prefetch_capacity,
                    dtypes=[tf.float32, tf.float32, tf.float32],
                    shapes=[
                        self.input_shape[1:], self.input_shape[1:],
                        self.input_shape[1:]
                    ])

                self.data_ph_from_queue = dict()
                self.data_ph_from_queue['anchor'] = None
                self.data_ph_from_queue['positive'] = None
                self.data_ph_from_queue['negative'] = None

                # Fetching the place holders from the queue
                self.enqueue_op = queue.enqueue_many([
                    self.data_ph['anchor'], self.data_ph['positive'],
                    self.data_ph['negative']
                ])
                self.data_ph_from_queue['anchor'], self.data_ph_from_queue[
                    'positive'], self.data_ph_from_queue[
                        'negative'] = queue.dequeue_many(self.batch_size)

            else:
                self.data_ph_from_queue = dict()
                self.data_ph_from_queue['anchor'] = self.data_ph['anchor']
                self.data_ph_from_queue['positive'] = self.data_ph['positive']
                self.data_ph_from_queue['negative'] = self.data_ph['negative']

    def get_triplets(self, input_data, input_labels):

        # Shuffling all the indexes
        indexes_per_labels = dict()
        for l in self.possible_labels:
            indexes_per_labels[l] = numpy.where(input_labels == l)[0]
            numpy.random.shuffle(indexes_per_labels[l])

        # searching for random triplets
        offset_class = 0

        for i in range(input_data.shape[0]):

            anchor = input_data[indexes_per_labels[self.possible_labels[
                offset_class]][numpy.random.randint(
                    len(indexes_per_labels[self.possible_labels[
                        offset_class]]))], ...]

            positive = input_data[indexes_per_labels[self.possible_labels[
                offset_class]][numpy.random.randint(
                    len(indexes_per_labels[self.possible_labels[
                        offset_class]]))], ...]

            # Changing the class
            offset_class += 1

            if offset_class == len(self.possible_labels):
                offset_class = 0

            negative = input_data[indexes_per_labels[self.possible_labels[
                offset_class]][numpy.random.randint(
                    len(indexes_per_labels[self.possible_labels[
                        offset_class]]))], ...]

            yield anchor, positive, negative

    def get_one_triplet(self, input_data, input_labels):
        # Getting a pair of clients
        index = numpy.random.choice(
            len(self.possible_labels), 2, replace=False)
        index[0] = self.possible_labels[index[0]]
        index[1] = self.possible_labels[index[1]]

        # Getting the indexes of the data from a particular client
        indexes = numpy.where(input_labels == index[0])[0]
        numpy.random.shuffle(indexes)

        # Picking a positive pair
        data_anchor = input_data[indexes[0], ...]
        data_positive = input_data[indexes[1], ...]

        # Picking a negative sample
        indexes = numpy.where(input_labels == index[1])[0]
        numpy.random.shuffle(indexes)
        data_negative = input_data[indexes[0], ...]

        return data_anchor, data_positive, data_negative
