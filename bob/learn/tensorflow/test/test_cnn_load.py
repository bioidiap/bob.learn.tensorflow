#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST


"""
Some unit tests that create networks on the fly
"""


import numpy
import pkg_resources
from bob.learn.tensorflow.utils import load_mnist
from bob.learn.tensorflow.network import SequenceNetwork
from bob.learn.tensorflow.datashuffler import Memory
import tensorflow as tf


def validate_network(validation_data, validation_labels, network):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[28, 28, 1],
                                      batch_size=400)

    [data, labels] = validation_data_shuffler.get_batch()
    predictions = network.predict(data)
    accuracy = 100. * numpy.sum(predictions == labels) / predictions.shape[0]

    return accuracy


def test_load_test_cnn():
    tf.reset_default_graph()

    _, _, validation_data, validation_labels = load_mnist()

    # Creating datashufflers
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))
    network = SequenceNetwork()
    network.load(pkg_resources.resource_filename(__name__, 'data/cnn_mnist/model.ckp'))

    accuracy = validate_network(validation_data, validation_labels, network)
    assert accuracy > 80
    del network

