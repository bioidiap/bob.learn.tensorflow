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
        """
        pass

    def __call__(self, data_shuffler, network, session):
        data, labels = data_shuffler.get_batch()

        predictions = numpy.argmax(session.run(network.inference_graph, feed_dict={network.inference_placeholder: data[:]}), 1)
        accuracy = 100. * numpy.sum(predictions == labels) / predictions.shape[0]

        summaries = [(summary_pb2.Summary.Value(tag="accuracy_validation", simple_value=float(accuracy)))]
        return summary_pb2.Summary(value=summaries)
