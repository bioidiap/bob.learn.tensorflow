#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST

import numpy
import tensorflow as tf

from .OnlineSampling import OnlineSampling
from .Memory import Memory
from .Triplet import Triplet
from scipy.spatial.distance import euclidean, cdist

import logging
logger = logging.getLogger("bob.learn")


class TripletWithSelectionMemory(Triplet, Memory, OnlineSampling):
    """
    This data shuffler generates triplets from :py:class:`bob.learn.tensorflow.datashuffler.Triplet` and
    :py:class:`bob.learn.tensorflow.datashuffler.Memory` shufflers.

    The selection of the triplets is inspired in the paper:

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition. 2015.

    In this shuffler, the triplets are selected as the following:

    1. Select M identities.
    2. Get N pairs anchor-positive (for each M identities) such that the argmax(anchor, positive).
    3. For each pair anchor-positive, find the "semi-hard" negative samples such that :math:`argmin(||f(x_a) - f(x_p)||^2 < ||f(x_a) - f(x_n)||^2`

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

    def __init__(self,
                 data,
                 labels,
                 input_shape,
                 input_dtype="float32",
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 total_identities=10,
                 normalizer=None):

        super(TripletWithSelectionMemory, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer)
        self.clear_variables()
        # Seting the seed
        numpy.random.seed(seed)

        self.data = self.data.astype(input_dtype)
        self.total_identities = total_identities
        self.first_batch = True
        self.batch_increase_factor = 4

    def get_batch(self):
        """
        Get SELECTED triplets

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        shape = [self.batch_size] + list(self.input_shape[1:])

        # Selecting the classes used in the selection
        indexes = numpy.random.choice(
            len(self.possible_labels), self.total_identities, replace=False)
        samples_per_identity = numpy.ceil(
            self.batch_size / float(self.total_identities))
        anchor_labels = numpy.ones(
            samples_per_identity) * self.possible_labels[indexes[0]]

        for i in range(1, self.total_identities):
            anchor_labels = numpy.hstack((anchor_labels,
                                          numpy.ones(samples_per_identity) *
                                          self.possible_labels[indexes[i]]))
        anchor_labels = anchor_labels[0:self.batch_size]

        samples_a = numpy.zeros(shape=shape, dtype=self.input_dtype)

        # Computing the embedding
        for i in range(shape[0]):
            samples_a[i, ...] = self.get_anchor(anchor_labels[i])
        embedding_a = self.project(samples_a)

        # Getting the positives
        samples_p, embedding_p, d_anchor_positive = self.get_positives(
            anchor_labels, embedding_a)
        samples_n = self.get_negative(anchor_labels, embedding_a,
                                      d_anchor_positive)

        return samples_a, samples_p, samples_n

    def get_anchor(self, label):
        """
        Select random samples as anchor
        """

        # Getting the indexes of the data from a particular client
        indexes = numpy.where(self.labels == label)[0]
        numpy.random.shuffle(indexes)

        return self.normalize_sample(self.data[indexes[0], ...])

    def get_positives(self, anchor_labels, embedding_a):
        """
        Get the a random set of positive pairs
        """
        shape = [self.batch_size] + list(self.input_shape[1:])

        samples_p = numpy.zeros(shape=shape, dtype='float32')
        for i in range(self.shape[0]):
            l = anchor_labels[i]
            indexes = numpy.where(self.labels == l)[0]
            numpy.random.shuffle(indexes)
            samples_p[i, ...] = self.normalize_sample(
                self.data[indexes[0], ...])

        embedding_p = self.project(samples_p)

        # Computing the distances
        d_anchor_positive = []
        for i in range(shape[0]):
            d_anchor_positive.append(
                euclidean(embedding_a[i, :], embedding_p[i, :]))

        return samples_p, embedding_p, d_anchor_positive

    def get_negative(self, anchor_labels, embedding_a, d_anchor_positive):
        """
        Get the the semi-hard negative
        """

        shape = [self.batch_size] + list(self.input_shape[1:])

        # Shuffling all the dataset
        indexes = range(len(self.labels))
        numpy.random.shuffle(indexes)

        negative_samples_search = self.batch_size * self.batch_increase_factor

        # Limiting to the batch size, otherwise the number of comparisons will explode
        indexes = indexes[0:negative_samples_search]

        # Loading samples for the semi-hard search
        temp_samples_n = numpy.zeros(shape=shape, dtype='float32')
        samples_n = numpy.zeros(shape=shape, dtype='float32')
        for i in range(shape[0]):
            temp_samples_n[i, ...] = self.normalize_sample(
                self.data[indexes[i], ...])

        # Computing all the embeddings
        embedding_temp_n = self.project(temp_samples_n)

        # Computing the distances
        d_anchor_negative = cdist(
            embedding_a, embedding_temp_n, metric='euclidean')

        # Selecting the negative samples
        for i in range(shape[0]):
            label = anchor_labels[i]
            possible_candidates = [
                d if d > d_anchor_positive[i] else numpy.inf
                for d in d_anchor_negative[i]
            ]

            for j in numpy.argsort(possible_candidates):

                # Checking if they don't have the same label
                if self.labels[indexes[j]] != label:
                    samples_n[i, ...] = temp_samples_n[j, ...]
                    if numpy.isinf(possible_candidates[j]):
                        logger.info(
                            "SEMI-HARD negative not found, took the first one")
                    break

        return samples_n
