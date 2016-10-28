#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
import bob.io.base
import os
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation
from bob.learn.tensorflow.initialization import Xavier, Constant
from bob.learn.tensorflow.network import SequenceNetwork
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer
from bob.learn.tensorflow.util import load_mnist
from bob.learn.tensorflow.layers import Conv2D, FullyConnected, MaxPooling
import tensorflow as tf
import shutil

"""
Some unit tests that create networks on the fly
"""

batch_size = 16
validation_batch_size = 400
iterations = 50
seed = 10


def scratch_network():
    # Creating a random network
    scratch = SequenceNetwork()
    scratch.add(Conv2D(name="conv1", kernel_size=3,
                       filters=10,
                       activation=tf.nn.tanh,
                       weights_initialization=Xavier(seed=seed, use_gpu=False),
                       bias_initialization=Constant(use_gpu=False)))
    scratch.add(FullyConnected(name="fc1", output_dim=10,
                               activation=None,
                               weights_initialization=Xavier(seed=seed, use_gpu=False),
                               bias_initialization=Constant(use_gpu=False)))

    return scratch


def validate_network(validation_data, validation_labels, directory):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[28, 28, 1],
                                      batch_size=validation_batch_size)
    with tf.Session() as session:

        validation_shape = [400, 28, 28, 1]
        scratch = SequenceNetwork()
        scratch.load(bob.io.base.HDF5File(os.path.join(directory, "model.hdf5")),
                     shape=validation_shape, session=session)
        [data, labels] = validation_data_shuffler.get_batch()
        predictions = scratch(data, session=session)
        accuracy = 100. * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0]

    return accuracy


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

    # Create scratch network
    scratch = scratch_network()

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(architecture=scratch,
                      loss=loss,
                      iterations=iterations,
                      analizer=None,
                      prefetch=False,
                      temp_dir=directory)
    trainer.train(train_data_shuffler)

    accuracy = validate_network(validation_data, validation_labels, directory)
    assert accuracy > 80
    del scratch


