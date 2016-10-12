#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:33 CEST

"""
Neural net work error rates analizer
"""
import numpy
import bob.measure
from tensorflow.core.framework import summary_pb2
from scipy.spatial.distance import cosine


class ExperimentAnalizer:
    """
    Analizer.

    I don't know if this is the best way to do, but what this class do is the following.

    As an enrollment sample, averare all the TRAINING samples for one particular class.
    The probing is done with the validation set

    """

    def __init__(self, convergence_threshold=0.01, convergence_reference='eer'):
        """
        Use the CNN as feature extractor for a n-class classification

        ** Parameters **

          data_shuffler:
          graph:
          session:
          convergence_threshold:
          convergence_reference: References to analize the convergence. Possible values are `eer`, `far10`, `far10`


        """

        self.data_shuffler = None
        self.network = None
        self.session = None

        # Statistics
        self.eer = []
        self.far10 = []
        self.far100 = []
        self.far1000 = []

    def __call__(self, data_shuffler, network, session):

        if self.data_shuffler is None:
            self.data_shuffler = data_shuffler
            self.network = network
            self.session = session

        # Extracting features for enrollment
        enroll_data, enroll_labels = self.data_shuffler.get_batch()
        enroll_features = self.network(enroll_data, session=self.session)
        del enroll_data

        # Extracting features for probing
        probe_data, probe_labels = self.data_shuffler.get_batch()
        probe_features = self.network(probe_data, session=self.session)
        del probe_data

        # Creating models
        models = []
        for i in range(len(self.data_shuffler.possible_labels)):
            indexes_model = numpy.where(enroll_labels == self.data_shuffler.possible_labels[i])[0]
            models.append(numpy.mean(enroll_features[indexes_model, :], axis=0))

        # Probing
        positive_scores = numpy.zeros(shape=0)
        negative_scores = numpy.zeros(shape=0)
        for i in range(len(self.data_shuffler.possible_labels)):
            #for i in self.data_shuffler.possible_labels:
            # Positive scoring
            indexes = probe_labels == self.data_shuffler.possible_labels[i]
            positive_data = probe_features[indexes, :]
            p = [cosine(models[i], positive_data[j]) for j in range(positive_data.shape[0])]
            positive_scores = numpy.hstack((positive_scores, p))

            # negative scoring
            indexes = probe_labels != self.data_shuffler.possible_labels[i]
            negative_data = probe_features[indexes, :]
            n = [cosine(models[i], negative_data[j]) for j in range(negative_data.shape[0])]
            negative_scores = numpy.hstack((negative_scores, n))

        return self.__compute_tensorflow_summary((-1)*negative_scores, (-1) * positive_scores)

    def __compute_tensorflow_summary(self, negative_scores, positive_scores):
        """
        Compute some stats with the scores, such as:
          - EER
          - FAR 10
          - FAR 100
          - FAR 1000
          - RANK 1
          - RANK 10

        **Parameters**
          negative_scores:
          positive_scores:
        """

        summaries = []

        # Compute EER
        threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        eer = (far + frr) / 2.
        summaries.append(summary_pb2.Summary.Value(tag="EER", simple_value=eer))
        self.eer.append(eer)

        # Computing FAR 10
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, far_value=0.1)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        summaries.append(summary_pb2.Summary.Value(tag="FAR 10", simple_value=frr))
        self.far10.append(frr)

        # Computing FAR 100
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, far_value=0.01)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        summaries.append(summary_pb2.Summary.Value(tag="FAR 100", simple_value=frr))
        self.far100.append(frr)

        # Computing FAR 1000
        threshold = bob.measure.far_threshold(negative_scores, positive_scores, far_value=0.001)
        far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
        summaries.append(summary_pb2.Summary.Value(tag="FAR 1000", simple_value=frr))
        self.far1000.append(frr)

        return summary_pb2.Summary(value=summaries)