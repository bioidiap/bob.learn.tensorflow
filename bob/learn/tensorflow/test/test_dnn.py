#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory, Disk, SiameseDisk, TripletDisk
from bob.learn.tensorflow.network import Chopra, MLP
from bob.learn.tensorflow.loss import BaseLoss, ContrastiveLoss, TripletLoss
from bob.learn.tensorflow.trainers import Trainer, SiameseTrainer, TripletTrainer
# from ..analyzers import ExperimentAnalizer, SoftmaxAnalizer
from bob.learn.tensorflow.util import load_mnist
import tensorflow as tf
import bob.io.base
import os
import shutil
from scipy.spatial.distance import cosine
import bob.measure

"""
Some unit tests for the datashuffler
"""

batch_size = 16
validation_batch_size = 400
iterations = 50
seed = 10


def test_dnn_trainer():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[28, 28, 1],
                                 batch_size=batch_size)

    with tf.Session() as session:
        directory = "./temp/dnn"

        # Preparing the architecture
        architecture = MLP(10, hidden_layers=[15, 20])

        # Loss for the softmax
        loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

        # One graph trainer
        trainer = Trainer(architecture=architecture,
                          loss=loss,
                          iterations=iterations,
                          analizer=None,
                          prefetch=False,
                          temp_dir=directory)
        trainer.train(train_data_shuffler)

        # Testing
        validation_shape = [400, 28, 28, 1]
        mlp = MLP(10, hidden_layers=[15, 20])
        mlp.load(bob.io.base.HDF5File(os.path.join(directory, "model.hdf5")),
                 shape=validation_shape, session=session)
        validation_data_shuffler = Memory(validation_data, validation_labels,
                                          input_shape=[28, 28, 1],
                                          batch_size=validation_batch_size)

        [data, labels] = validation_data_shuffler.get_batch()
        predictions = mlp(data, session=session)
        accuracy = 100. * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0]

        # At least 50% of accuracy for the DNN
        assert accuracy > 50.
        shutil.rmtree(directory)
        session.close()