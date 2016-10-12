#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:33 CEST

"""
Neural net work error rates analizer
"""
import numpy
from tensorflow.core.framework import summary_pb2


class SoftmaxAnalizer(object):
    """
    Analizer.
    """

    def __init__(self):
        """
        Softmax analizer

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

    def __call__(self, data_shuffler, network, session):

        if self.data_shuffler is None:
            self.data_shuffler = data_shuffler
            self.network = network
            self.session = session

        # Creating the graph
        feature_batch, label_batch = self.data_shuffler.get_placeholders(name="validation_accuracy")
        data, labels = self.data_shuffler.get_batch()
        graph = self.network.compute_graph(feature_batch)

        predictions = numpy.argmax(self.session.run(graph, feed_dict={feature_batch: data[:]}), 1)
        accuracy = 100. * numpy.sum(predictions == labels) / predictions.shape[0]

        summaries = []
        summaries.append(summary_pb2.Summary.Value(tag="accuracy_validation", simple_value=float(accuracy)))
        return summary_pb2.Summary(value=summaries)
