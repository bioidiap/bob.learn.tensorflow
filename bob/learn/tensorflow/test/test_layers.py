#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import tensorflow as tf
import numpy as np
from bob.learn.tensorflow.layers import maxout
from nose.tools import assert_raises_regexp

slim = tf.contrib.slim

def test_simple():
    x = np.zeros([64, 10, 36])
    graph = maxout(x, num_units=3)
    assert graph.get_shape().as_list() == [64, 10, 3]

def test_fully_connected():
    x = np.zeros([64, 50])
    graph = slim.fully_connected(x, 50, activation_fn=None)
    graph = maxout(graph, num_units=10)
    assert graph.get_shape().as_list() == [64, 10]

def test_nchw():
    x = np.random.uniform(size=(10, 100, 100, 3)).astype(np.float32)
    graph = slim.conv2d(x, 10, [3, 3])
    graph = maxout(graph, num_units=1)
    assert graph.get_shape().as_list() == [10, 100, 100, 1]

def test_invalid_shape():
    x = np.random.uniform(size=(10, 100, 100, 3)).astype(np.float32)
    graph = slim.conv2d(x, 3, [3, 3])
    with assert_raises_regexp(ValueError, 'number of features'):
        graph = maxout(graph, num_units=2)

