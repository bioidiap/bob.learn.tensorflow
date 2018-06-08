#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.learn.tensorflow.network import inception_resnet_v2, inception_resnet_v2_batch_norm,\
                                         inception_resnet_v1, inception_resnet_v1_batch_norm


def test_inceptionv2():

    # Testing WITHOUT batch norm
    inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
    graph, _ = inception_resnet_v2(inputs)
    assert len(tf.trainable_variables()) == 490

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Testing WITH batch norm
    inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
    graph, _ = inception_resnet_v2_batch_norm(inputs)
    assert len(tf.trainable_variables()) == 900

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


def test_inceptionv1():

    # Testing WITHOUT batch norm
    inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
    graph, _ = inception_resnet_v1(inputs)
    assert len(tf.trainable_variables()) == 266

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Testing WITH batch norm
    inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
    graph, _ = inception_resnet_v1_batch_norm(inputs)
    assert len(tf.trainable_variables()) == 490

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
