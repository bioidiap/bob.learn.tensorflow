#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from nose.plugins.attrib import attr
import tensorflow as tf
from bob.learn.tensorflow.network import inception_resnet_v2, inception_resnet_v2_batch_norm,\
    inception_resnet_v1, inception_resnet_v1_batch_norm,\
    vgg_19, vgg_16, mlp_with_batchnorm_and_dropout

# @attr('slow')
# def test_inceptionv2():

#     tf.reset_default_graph()
#     # Testing WITHOUT batch norm
#     inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
#     graph, _ = inception_resnet_v2(inputs)
#     assert len(tf.trainable_variables()) == 490

#     tf.reset_default_graph()
#     assert len(tf.global_variables()) == 0

#     # Testing WITH batch norm
#     inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
#     graph, _ = inception_resnet_v2_batch_norm(inputs)
#     assert len(tf.trainable_variables()) == 490, len(tf.trainable_variables())

#     tf.reset_default_graph()
#     assert len(tf.global_variables()) == 0

# @attr('slow')
# def test_inceptionv2_adaptation():

#     tf.reset_default_graph()
#     for n, trainable_variables in [
#         (490, None),
#         (0, []),
#         (2, ['Conv2d_1a_3x3', 'Conv2d_1a_3x3_BN']),
#         (4, ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3_BN',
#              'Conv2d_2a_3x3_BN']),
#         (6, ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
#              'Conv2d_1a_3x3_BN', 'Conv2d_2a_3x3_BN', 'Conv2d_2b_3x3_BN']),
#         (1, ['Conv2d_1a_3x3_BN']),
#         (2, ['Conv2d_1a_3x3_BN', 'Conv2d_2a_3x3_BN']),
#         (3, ['Conv2d_1a_3x3_BN', 'Conv2d_2a_3x3_BN', 'Conv2d_2b_3x3_BN']),
#     ]:
#         input = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
#         net, end_points = inception_resnet_v2_batch_norm(
#             input, trainable_variables=trainable_variables)
#         l = len(tf.trainable_variables())
#         assert l == n, (l, n)
#         tf.reset_default_graph()
#     tf.reset_default_graph()
#     assert len(tf.global_variables()) == 0

# @attr('slow')
# def test_inceptionv1():

#     tf.reset_default_graph()
#     # Testing WITHOUT batch norm
#     inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
#     graph, _ = inception_resnet_v1(inputs)
#     assert len(tf.trainable_variables()) == 266

#     tf.reset_default_graph()
#     assert len(tf.global_variables()) == 0

#     # Testing WITH batch norm
#     inputs = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
#     graph, _ = inception_resnet_v1_batch_norm(inputs)
#     assert len(tf.trainable_variables()) == 266

#     tf.reset_default_graph()
#     assert len(tf.global_variables()) == 0

# @attr('slow')
# def test_inceptionv1_adaptation():

#     tf.reset_default_graph()
#     for n, trainable_variables in [
#         (266, None),
#         (0, []),
#         (2, ['Conv2d_1a_3x3', 'Conv2d_1a_3x3_BN']),
#         (4, ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_1a_3x3_BN',
#              'Conv2d_2a_3x3_BN']),
#         (6, ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
#              'Conv2d_1a_3x3_BN', 'Conv2d_2a_3x3_BN', 'Conv2d_2b_3x3_BN']),
#         (1, ['Conv2d_1a_3x3_BN']),
#         (2, ['Conv2d_1a_3x3_BN', 'Conv2d_2a_3x3_BN']),
#         (3, ['Conv2d_1a_3x3_BN', 'Conv2d_2a_3x3_BN', 'Conv2d_2b_3x3_BN']),
#     ]:
#         input = tf.placeholder(tf.float32, shape=(1, 160, 160, 1))
#         net, end_points = inception_resnet_v1_batch_norm(
#             input, trainable_variables=trainable_variables)
#         l = len(tf.trainable_variables())
#         assert l == n, (l, n)
#         tf.reset_default_graph()
#     tf.reset_default_graph()
#     assert len(tf.global_variables()) == 0


def test_vgg():
    tf.reset_default_graph()

    # Testing VGG19 Training mode
    inputs = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    graph, _ = vgg_19(inputs)
    assert len(tf.trainable_variables()) == 38

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Testing VGG19 predicting mode
    inputs = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    graph, _ = vgg_19(inputs, mode=tf.estimator.ModeKeys.PREDICT)
    assert len(tf.trainable_variables()) == 0

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


    # Testing VGG 16 training mode
    inputs = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    graph, _ = vgg_16(inputs)
    assert len(tf.trainable_variables()) == 30

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Testing VGG 16 predicting mode
    inputs = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    graph, _ = vgg_16(inputs, mode=tf.estimator.ModeKeys.PREDICT)
    assert len(tf.trainable_variables()) == 0

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


def test_mlp():

    tf.reset_default_graph()
    # Testing MLP Training mode
    inputs = tf.placeholder(tf.float32, shape=(1, 10, 10, 3))
    graph, _ = mlp_with_batchnorm_and_dropout(inputs, [6, 5])
    assert len(tf.trainable_variables()) == 4

    tf.reset_default_graph()
    # Testing MLP Predicting mode
    inputs = tf.placeholder(tf.float32, shape=(1, 10, 10, 3))
    graph, _ = mlp_with_batchnorm_and_dropout(inputs, [6, 5], mode=tf.estimator.ModeKeys.PREDICT)
    assert len(tf.trainable_variables()) == 0

