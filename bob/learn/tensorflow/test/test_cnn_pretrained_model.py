#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
import bob.io.base
import os
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
from bob.learn.tensorflow.network import SequenceNetwork
from bob.learn.tensorflow.layers import Conv2D, FullyConnected

import tensorflow as tf
import shutil

"""
Some unit tests that create networks on the fly and load variables
"""

batch_size = 16
validation_batch_size = 400
iterations = 50
seed = 10


def scratch_network():
    # Creating a random network
    scratch = SequenceNetwork(default_feature_layer="fc1")
    scratch.add(Conv2D(name="conv1", kernel_size=3,
                       filters=10,
                       activation=tf.nn.tanh,
                       batch_norm=False))
    scratch.add(FullyConnected(name="fc1", output_dim=10,
                               activation=None,
                               batch_norm=False
                               ))

    return scratch


def validate_network(validation_data, validation_labels, network):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[28, 28, 1],
                                      batch_size=validation_batch_size)

    [data, labels] = validation_data_shuffler.get_batch()
    predictions = network.predict(data)
    accuracy = 100. * numpy.sum(predictions == labels) / predictions.shape[0]

    return accuracy


def test_cnn_pretrained():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[28, 28, 1],
                                 batch_size=batch_size,
                                 data_augmentation=data_augmentation)
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    directory = "./temp/cnn"
    directory2 = "./temp/cnn2"

    # Creating a random network
    scratch = scratch_network()

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(architecture=scratch,
                      loss=loss,
                      iterations=iterations,
                      analizer=None,
                      prefetch=False,
                      learning_rate=constant(0.05, name="lr"),
                      temp_dir=directory)
    trainer.train(train_data_shuffler)
    accuracy = validate_network(validation_data, validation_labels, scratch)
    assert accuracy > 85

    del scratch
    del loss
    del trainer

    # Training the network using a pre trained model
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean, name="loss")
    scratch = scratch_network()
    trainer = Trainer(architecture=scratch,
                      loss=loss,
                      iterations=iterations+200,
                      analizer=None,
                      prefetch=False,
                      learning_rate=constant(0.05, name="lr2"),
                      temp_dir=directory2,
                      model_from_file=os.path.join(directory, "model.ckp"))

    trainer.train(train_data_shuffler)

    accuracy = validate_network(validation_data, validation_labels, scratch)
    assert accuracy > 90
    shutil.rmtree(directory)
    shutil.rmtree(directory2)

    del scratch
    del loss
    del trainer

