#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor
from bob.learn.tensorflow.network import Embedding
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import shutil

"""
Some unit tests that create networks on the fly
"""

batch_size = 16
validation_batch_size = 400
iterations = 300
seed = 10
directory = "./temp/cnn_scratch"

slim = tf.contrib.slim


def scratch_network():
    # Creating a random network
    inputs = dict()
    inputs['data'] = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="train_data")
    inputs['label'] = tf.placeholder(tf.int64, shape=[None], name="train_label")

    initializer = tf.contrib.layers.xavier_initializer(seed=seed)
    graph = slim.conv2d(inputs['data'], 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                        weights_initializer=initializer)
    graph = slim.max_pool2d(graph, [4, 4], scope='pool1')
    graph = slim.flatten(graph, scope='flatten1')
    graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer)

    return inputs, graph


def validate_network(embedding, validation_data, validation_labels):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[None, 28, 28, 1],
                                      batch_size=validation_batch_size,
                                      normalizer=ScaleFactor())

    [data, labels] = validation_data_shuffler.get_batch()
    predictions = embedding(data)
    accuracy = 100. * numpy.sum(numpy.argmax(predictions, axis=1) == labels) / predictions.shape[0]

    return accuracy


def test_cnn_trainer_scratch():

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 28, 28, 1],
                                 batch_size=batch_size,
                                 data_augmentation=data_augmentation,
                                 normalizer=ScaleFactor())

    #validation_data_shuffler = Memory(train_data, train_labels,
    #                                  input_shape=[28, 28, 1],
    #                                  batch_size=batch_size,
    #                                  data_augmentation=data_augmentation,
    #                                  normalizer=ScaleFactor())

    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))
    # Create scratch network
    inputs, scratch = scratch_network()

    # Setting the placeholders
    train_data_shuffler.data_ph = inputs['data']
    train_data_shuffler.label_ph = inputs['label']
    embedding = Embedding(inputs['data'], scratch)

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)

    trainer.create_network_from_scratch(graph=scratch,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),
                                        )

    trainer.train()
    accuracy = validate_network(embedding, validation_data, validation_labels)
    assert accuracy > 70
    shutil.rmtree(directory)
    del trainer
