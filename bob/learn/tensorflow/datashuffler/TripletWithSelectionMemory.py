#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

from .Memory import Memory
from .Triplet import Triplet
from .OnlineSampling import OnLineSampling
from scipy.spatial.distance import euclidean
from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class TripletWithSelectionMemory(Triplet, Memory, OnLineSampling):
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
                 total_identities=10,
                 normalizer=Linear()):

        super(TripletWithSelectionMemory, self).__init__(
            data=data,
            labels=labels,
            input_shape=input_shape,
            input_dtype=input_dtype,
            scale=scale,
            batch_size=batch_size,
            seed=seed,
            data_augmentation=data_augmentation,
            normalizer=normalizer
        )
        self.clear_variables()
        # Seting the seed
        numpy.random.seed(seed)

        self.data = self.data.astype(input_dtype)
        self.total_identities = total_identities
        self.first_batch = True


    def get_random_batch(self):
        """
        Get a random triplet

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        for i in range(self.shape[0]):
            data_a[i, ...], data_p[i, ...], data_n[i, ...] = self.get_one_triplet(self.data, self.labels)

        return [data_a, data_p, data_n]

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
        samples_per_identity = self.batch_size/self.total_identities
        anchor_labels = numpy.ones(samples_per_identity) * indexes[0]
        for i in range(1, self.total_identities):
            anchor_labels = numpy.hstack((anchor_labels,numpy.ones(samples_per_identity) * indexes[i]))
        anchor_labels = anchor_labels[0:self.batch_size]

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        # Fetching the anchors
        for i in range(self.shape[0]):
            data_a[i, ...] = self.get_anchor(anchor_labels[i])
        features_a = self.project(data_a)

        for i in range(self.shape[0]):
            label = anchor_labels[i]
            #anchor = self.get_anchor(label)
            positive, distance_anchor_positive = self.get_positive(label, features_a[i])
            negative = self.get_negative(label, features_a[i], distance_anchor_positive)

            data_p[i, ...] = positive
            data_n[i, ...] = negative

        # Applying the data augmentation
        if self.data_augmentation is not None:
            for i in range(data_a.shape[0]):
                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_a[i, ...])))
                data_a[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_p[i, ...])))
                data_p[i, ...] = d

                d = self.bob2skimage(self.data_augmentation(self.skimage2bob(data_n[i, ...])))
                data_n[i, ...] = d

        # Scaling
        data_a = self.normalize_sample(data_a)
        data_p = self.normalize_sample(data_p)
        data_n = self.normalize_sample(data_n)

        return data_a, data_p, data_n

    def get_anchor(self, label):
        """
        Select random samples as anchor
        """

        # Getting the indexes of the data from a particular client
        indexes = numpy.where(self.labels == label)[0]
        numpy.random.shuffle(indexes)

        return self.data[indexes[0], ...]

    def get_positive(self, label, anchor_feature):
        """
        Get the best positive sample given the anchor.
        The best positive sample for the anchor is the farthest from the anchor
        """

        # Projecting the anchor
        #anchor_feature = self.feature_extractor(self.reshape_for_deploy(anchor), session=self.session)

        indexes = numpy.where(self.labels == label)[0]
        numpy.random.shuffle(indexes)
        indexes = indexes[
                  0:self.batch_size]  # Limiting to the batch size, otherwise the number of comparisons will explode
        distances = []
        positive_features = self.project(self.data[indexes, ...])

        # Projecting the positive instances
        for p in positive_features:
            distances.append(euclidean(anchor_feature, p))

        # Geting the max
        index = numpy.argmax(distances)
        return self.data[indexes[index], ...], distances[index]

    def get_negative(self, label, anchor_feature, distance_anchor_positive):
        """
        Get the best negative sample for a pair anchor-positive
        """
        # Projecting the anchor
        #anchor_feature = self.feature_extractor(self.reshape_for_deploy(anchor), session=self.session)

        # Selecting the negative samples
        indexes = numpy.where(self.labels != label)[0]
        numpy.random.shuffle(indexes)
        indexes = indexes[
                  0:self.batch_size] # Limiting to the batch size, otherwise the number of comparisons will explode
        negative_features = self.project(self.data[indexes, ...])

        distances = []
        for n in negative_features:
            d = euclidean(anchor_feature, n)

            # Semi-hard samples criteria
            if d > distance_anchor_positive:
                distances.append(d)
            else:
                distances.append(numpy.inf)

        # Getting the minimum negative sample as the reference for the pair
        index = numpy.argmin(distances)

        # if the semi-hardest is inf take the first
        if numpy.isinf(distances[index]):
            index = 0

        return self.data[indexes[index], ...]
