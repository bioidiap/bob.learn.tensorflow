#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf

from .Disk import Disk
from .Triplet import Triplet
from .OnlineSampling import OnLineSampling
from scipy.spatial.distance import euclidean
from bob.learn.tensorflow.datashuffler.Normalizer import Linear

import logging
logger = logging.getLogger("bob.learn.tensorflow")
from bob.learn.tensorflow.datashuffler.Normalizer import Linear

class TripletWithSelectionDisk(Triplet, Disk, OnLineSampling):
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

        super(TripletWithSelectionDisk, self).__init__(
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

        self.total_identities = total_identities
        self.first_batch = True

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

        data_a = numpy.zeros(shape=self.shape, dtype='float32')
        data_p = numpy.zeros(shape=self.shape, dtype='float32')
        data_n = numpy.zeros(shape=self.shape, dtype='float32')

        #logger.info("Fetching anchor")
        # Fetching the anchors
        for i in range(self.shape[0]):
            data_a[i, ...] = self.get_anchor(anchor_labels[i])
        features_a = self.project(data_a)

        for i in range(self.shape[0]):
            #logger.info("*********Anchor {0}".format(i))

            label = anchor_labels[i]
            #anchor = self.get_anchor(label)
            #logger.info("********* Positives")
            positive, distance_anchor_positive = self.get_positive(label, features_a[i])
            #logger.info("********* Negatives")
            negative = self.get_negative(label, features_a[i], distance_anchor_positive)

            #logger.info("********* Appending")

            data_p[i, ...] = positive
            data_n[i, ...] = negative

        #logger.info("#################")
        # Scaling
        #if self.scale:
        #    data_a *= self.scale_value
        #    data_p *= self.scale_value
        #    data_n *= self.scale_value

        return data_a, data_p, data_n

    def get_anchor(self, label):
        """
        Select random samples as anchor
        """

        # Getting the indexes of the data from a particular client
        indexes = numpy.where(self.labels == label)[0]
        numpy.random.shuffle(indexes)

        file_name = self.data[indexes[0], ...]
        sample_a = self.load_from_file(str(file_name))

        sample_a = self.normalize_sample(sample_a)
        return sample_a

    def get_positive(self, label, embedding_a):
        """
        Get the best positive sample given the anchor.
        The best positive sample for the anchor is the farthest from the anchor
        """

        #logger.info("****************** numpy.where")
        indexes = numpy.where(self.labels == label)[0]

        numpy.random.shuffle(indexes)
        indexes = indexes[
                  0:self.batch_size]  # Limiting to the batch size, otherwise the number of comparisons will explode
        distances = []
        shape = tuple([len(indexes)] + list(self.shape[1:]))
        sample_p = numpy.zeros(shape=shape, dtype='float32')
        #logger.info("****************** search")
        for i in range(shape[0]):
            #logger.info("****************** fetch")
            file_name = self.data[indexes[i], ...]
            #logger.info("****************** load")
            sample_p[i, ...] = self.load_from_file(str(file_name))

        sample_p = self.normalize_sample(sample_p)

        #logger.info("****************** project")
        embedding_p = self.project(sample_p)

        #logger.info("****************** distances")
        # Projecting the positive instances
        for p in embedding_p:
            distances.append(euclidean(embedding_a, p))

        # Geting the max
        index = numpy.argmax(distances)
        #logger.info("****************** return")
        return sample_p[index, ...], distances[index]

    def get_negative(self, label, embedding_a, distance_anchor_positive):
        """
        Get the best negative sample for a pair anchor-positive
        """
        # Projecting the anchor
        #anchor_feature = self.feature_extractor(self.reshape_for_deploy(anchor), session=self.session)

        # Selecting the negative samples
        #logger.info("****************** numpy.where")
        indexes = numpy.where(self.labels != label)[0]
        numpy.random.shuffle(indexes)
        indexes = indexes[
                  0:self.batch_size*3] # Limiting to the batch size, otherwise the number of comparisons will explode

        shape = tuple([len(indexes)] + list(self.shape[1:]))
        sample_n = numpy.zeros(shape=shape, dtype='float32')
        #logger.info("****************** search")
        for i in range(shape[0]):
            #logger.info("****************** fetch")
            file_name = self.data[indexes[i], ...]
            #logger.info("****************** load")
            sample_n[i, ...] = self.load_from_file(str(file_name))

        sample_n = self.normalize_sample(sample_n)

        #logger.info("****************** project")
        embedding_n = self.project(sample_n)

        distances = []
        #logger.info("****************** distances")
        for n in embedding_n:
            d = euclidean(embedding_a, n)

            # Semi-hard samples criteria
            if d > distance_anchor_positive:
                distances.append(d)
            else:
                distances.append(numpy.inf)

        # Getting the minimum negative sample as the reference for the pair
        index = numpy.argmin(distances)

        # if the semi-hardest is inf take the first
        if numpy.isinf(distances[index]):
            logger.info("SEMI-HARD negative not found, took the first one")
            index = 0
        #logger.info("****************** return")
        return sample_n[index, ...]
