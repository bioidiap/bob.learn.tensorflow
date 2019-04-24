#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
import numpy
from bob.learn.tensorflow.loss import balanced_softmax_cross_entropy_loss_weights,\
                                      balanced_sigmoid_cross_entropy_loss_weights


def test_balanced_softmax_cross_entropy_loss_weights():
    labels = numpy.array([[1, 0, 0],
                          [1, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [1, 0, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1],
                          [0, 0, 1],
                          [1, 0, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype="int32")

    with tf.Session() as session:
        weights = session.run(balanced_softmax_cross_entropy_loss_weights(labels))
 
    expected_weights = numpy.array([0.53333336, 0.53333336, 1.5238096 , 2.1333334,\
                                    1.5238096 , 0.53333336, 0.53333336, 1.5238096,\
                                    0.53333336, 0.53333336, 0.53333336, 0.53333336,\
                                    0.53333336, 0.53333336, 2.1333334 , 0.53333336,\
                                    2.1333334 , 0.53333336, 1.5238096 , 1.5238096 ,\
                                    0.53333336, 1.5238096 , 0.53333336, 0.53333336,\
                                    2.1333334 , 0.53333336, 0.53333336, 0.53333336,\
                                    2.1333334 , 0.53333336, 1.5238096 , 0.53333336],\
                                    dtype="float32")

    assert numpy.allclose(weights, expected_weights)


def test_balanced_sigmoid_cross_entropy_loss_weights():
    labels = numpy.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
                          1, 1, 0, 1, 1, 1, 0, 1, 0, 1], dtype="int32")
    
    with tf.Session() as session:
        weights = session.run(balanced_sigmoid_cross_entropy_loss_weights(labels, dtype='float32'))
        
    expected_weights = numpy.array([0.8, 0.8, 1.3333334, 1.3333334, 1.3333334, 0.8,
                                    0.8, 1.3333334, 0.8, 0.8, 0.8, 0.8,
                                    0.8, 0.8, 1.3333334, 0.8, 1.3333334, 0.8,
                                    1.3333334, 1.3333334, 0.8, 1.3333334, 0.8, 0.8,
                                    1.3333334, 0.8, 0.8, 0.8, 1.3333334, 0.8,
                                    1.3333334, 0.8], dtype="float32")

    assert numpy.allclose(weights, expected_weights)

