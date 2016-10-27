#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

from .Disk import Disk
from .Triplet import Triplet
from .OnlineSampling import OnLineSampling
from scipy.spatial.distance import euclidean, cdist

import logging
logger = logging.getLogger("bob.learn.tensorflow")


class TripletWithFastSelectionDisk(Triplet, Disk, OnLineSampling):
    """
    This data shuffler generates triplets from :py:class:`bob.learn.tensorflow.datashuffler.Memory` shufflers.

    The selection of the triplets is inspired in the paper:

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition. 2015.


    In this shuffler, the triplets are selected as the following:
      1. Select M identities
      2. Get N pairs anchor-positive (for each M identities) such that the argmax(anchor, positive)
      3. For each pair anchor-positive, find the "semi-hard" negative samples such that
      argmin(||f(x_a) - f(x_p)||^2 < ||f(x_a) - f(x_n)||^2

     **Parameters**
       data:
       labels:
       perc_train:
       scale:
       train_batch_size:
       validation_batch_size:
       data_augmentation:
       total_identities: Number of identities inside of the batch
    """

    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10,
                 data_augmentation=None,
                 total_identities=10):

        super(TripletWithFastSelectionDisk, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation
        )
        self.clear_variables()
        # Seting the seed
        numpy.random.seed(seed)

        self.total_identities = total_identities
        self.first_batch = True

        # For the negative search I'll load `N` times the batch
        self.batch_increase_factor = 4

    def get_random_batch(self):
        """
        Get a random triplet

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        sample_a = numpy.zeros(shape=self.shape, dtype='float32')
        sample_p = numpy.zeros(shape=self.shape, dtype='float32')
        sample_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            file_name_a, file_name_p, file_name_n = self.get_one_triplet(self.data, self.labels)
            sample_a[i, ...] = self.load_from_file(str(file_name_a))
            sample_p[i, ...] = self.load_from_file(str(file_name_p))
            sample_n[i, ...] = self.load_from_file(str(file_name_n))

        sample_a = self.normalize_sample(sample_a)
        sample_p = self.normalize_sample(sample_p)
        sample_n = self.normalize_sample(sample_n)

        return [sample_a, sample_p, sample_n]

    def get_batch(self):
        """
        Get SELECTED triplets

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        if self.first_batch:
            self.first_batch = False
            return self.get_random_batch()

        # Selecting the classes used in the selection
        indexes = numpy.random.choice(len(self.possible_labels), self.total_identities, replace=False)
        samples_per_identity = numpy.ceil(self.batch_size/float(self.total_identities))
        anchor_labels = numpy.ones(samples_per_identity) * self.possible_labels[indexes[0]]

        for i in range(1, self.total_identities):
            anchor_labels = numpy.hstack((anchor_labels,numpy.ones(samples_per_identity) * self.possible_labels[indexes[i]]))
        anchor_labels = anchor_labels[0:self.batch_size]

        samples_a = numpy.zeros(shape=self.shape, dtype='float32')

        # Computing the embedding
        for i in range(self.shape[0]):
            samples_a[i, ...] = self.get_anchor(anchor_labels[i])
        embedding_a = self.project(samples_a)

        print "EMBEDDING {0} ".format(embedding_a[:, 0])

        # Getting the positives
        samples_p, embedding_p, d_anchor_positive = self.get_positives(anchor_labels, embedding_a)
        samples_n = self.get_negative(anchor_labels, embedding_a, d_anchor_positive)

        return samples_a, samples_p, samples_n

    def get_anchor(self, label):
        """
        Select random samples as anchor
        """

        # Getting the indexes of the data from a particular client
        indexes = numpy.where(self.labels == label)[0]
        numpy.random.shuffle(indexes)

        file_name = self.data[indexes[0], ...]
        anchor = self.load_from_file(str(file_name))

        anchor = self.normalize_sample(anchor)

        return anchor

    def get_positives(self, anchor_labels, embedding_a):
        """
        Get the a random set of positive pairs
        """
        samples_p = numpy.zeros(shape=self.shape, dtype='float32')
        for i in range(self.shape[0]):
            l = anchor_labels[i]
            indexes = numpy.where(self.labels == l)[0]
            numpy.random.shuffle(indexes)
            file_name = self.data[indexes[0], ...]
            samples_p[i, ...] = self.load_from_file(str(file_name))

        samples_p = self.normalize_sample(samples_p)
        embedding_p = self.project(samples_p)

        # Computing the distances
        d_anchor_positive = []
        for i in range(self.shape[0]):
            d_anchor_positive.append(euclidean(embedding_a[i, :], embedding_p[i, :]))

        return samples_p, embedding_p, d_anchor_positive

    def get_negative(self, anchor_labels, embedding_a, d_anchor_positive):
        """
        Get the the semi-hard negative
        """

        # Shuffling all the dataset
        indexes = range(len(self.labels))
        numpy.random.shuffle(indexes)

        negative_samples_search = self.batch_size*self.batch_increase_factor

        # Limiting to the batch size, otherwise the number of comparisons will explode
        indexes = indexes[0:negative_samples_search]

        # Loading samples for the semi-hard search
        shape = tuple([len(indexes)] + list(self.shape[1:]))
        temp_samples_n = numpy.zeros(shape=shape, dtype='float32')
        samples_n = numpy.zeros(shape=self.shape, dtype='float32')
        for i in range(shape[0]):
            file_name = self.data[indexes[i], ...]
            temp_samples_n[i, ...] = self.load_from_file(str(file_name))
        temp_samples_n = self.normalize_sample(temp_samples_n)

        # Computing all the embeddings
        embedding_temp_n = self.project(temp_samples_n)

        # Computing the distances
        d_anchor_negative = cdist(embedding_a, embedding_temp_n, metric='euclidean')

        # Selecting the negative samples
        for i in range(self.shape[0]):
            label = anchor_labels[i]
            possible_candidates = [d if d > d_anchor_positive[i] else numpy.inf for d in  d_anchor_negative[i]]

            for j in numpy.argsort(possible_candidates):

                # Checking if they don't have the same label
                if indexes[j] != label:
                    samples_n[i, ...] = temp_samples_n[j, ...]
                    if numpy.isinf(possible_candidates[j]):
                        logger.info("SEMI-HARD negative not found, took the first one")
                    break

        return samples_n
