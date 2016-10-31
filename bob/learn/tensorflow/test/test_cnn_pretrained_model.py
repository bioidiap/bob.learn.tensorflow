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
from bob.learn.tensorflow.util import load_mnist
import tensorflow as tf
import shutil

"""
Some unit tests that create networks on the fly and load variables
"""

batch_size = 16
validation_batch_size = 400
iterations = 50
seed = 10

from test_cnn_scratch import scratch_network, validate_network


def test_cnn_trainer_scratch():
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

    accuracy = validate_network(validation_data, validation_labels, directory)
    assert accuracy > 85

    del scratch
    del loss
    # Training the network using a pre trained model
    loss2 = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean, name="loss2")
    scratch = scratch_network()
    trainer2 = Trainer(architecture=scratch,
                       loss=loss2,
                       iterations=iterations,
                       analizer=None,
                       prefetch=False,
                       learning_rate=constant(0.05, name="lr2"),
                       temp_dir=directory2,
                       model_from_file=os.path.join(directory, "model.hdf5"))

    trainer2.train(train_data_shuffler)
    accuracy = validate_network(validation_data, validation_labels, directory)
    assert accuracy > 90
