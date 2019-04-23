#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import numpy
from bob.learn.tensorflow.utils import compute_embedding_accuracy, \
     compute_embedding_accuracy_tensors

import tensorflow as tf
"""
Some unit tests for the datashuffler
"""


def test_embedding_accuracy():

    numpy.random.seed(10)
    samples_per_class = 5

    class_a = numpy.random.normal(
        loc=0, scale=0.1, size=(samples_per_class, 2))
    labels_a = numpy.zeros(samples_per_class)

    class_b = numpy.random.normal(
        loc=10, scale=0.1, size=(samples_per_class, 2))
    labels_b = numpy.ones(samples_per_class)

    data = numpy.vstack((class_a, class_b))
    labels = numpy.concatenate((labels_a, labels_b))

    assert compute_embedding_accuracy(data, labels) == 1.

    # Adding noise
    noise = numpy.random.normal(loc=0, scale=0.1, size=(samples_per_class, 2))
    noise_labels = numpy.ones(samples_per_class)

    data = numpy.vstack((data, noise))
    labels = numpy.concatenate((labels, noise_labels))

    assert compute_embedding_accuracy(data, labels) == 10 / 15.


def test_embedding_accuracy_tensors():

    numpy.random.seed(10)
    samples_per_class = 5

    class_a = numpy.random.normal(
        loc=0, scale=0.1, size=(samples_per_class, 2))
    labels_a = numpy.zeros(samples_per_class)

    class_b = numpy.random.normal(
        loc=10, scale=0.1, size=(samples_per_class, 2))
    labels_b = numpy.ones(samples_per_class)

    data = numpy.vstack((class_a, class_b))
    labels = numpy.concatenate((labels_a, labels_b))

    data = tf.convert_to_tensor(data.astype("float32"))
    labels = tf.convert_to_tensor(labels.astype("int64"))

    sess = tf.Session()
    accuracy = sess.run(compute_embedding_accuracy_tensors(data, labels))
    assert accuracy == 1.
