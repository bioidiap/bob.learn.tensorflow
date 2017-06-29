#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Disk, ScaleFactor, TripletDisk
from bob.learn.tensorflow.loss import BaseLoss, ContrastiveLoss, TripletLoss
from bob.learn.tensorflow.trainers import Trainer, SiameseTrainer, TripletTrainer, constant
import shutil
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception

from test_datashuffler import get_dummy_files

"""
Some unit tests for the datashuffler
"""

iterations = 5
seed = 10


def test_inception_trainer():
    directory = "./temp/inception"

    # Loading data
    train_data, train_labels = get_dummy_files()
    batch_shape = [None, 224, 224, 3]

    train_data_shuffler = Disk(train_data, train_labels,
                               input_shape=batch_shape,
                               batch_size=2)

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # Creating inception model
    inputs = train_data_shuffler("data", from_queue=False)
    graph = inception.inception_v1(inputs)[0]

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory
                      )
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),
                                        )
    trainer.train()
    shutil.rmtree(directory)


def test_inception_triplet_trainer():
    directory = "./temp/inception"

    # Loading data
    train_data, train_labels = get_dummy_files()
    batch_shape = [None, 224, 224, 3]

    train_data_shuffler = TripletDisk(train_data, train_labels,
                                      input_shape=batch_shape,
                                      batch_size=2)

    # Loss for the softmax
    loss = TripletLoss()

    # Creating inception model
    inputs = train_data_shuffler("data", from_queue=False)

    graph = dict()
    graph['anchor'] = inception.inception_v1(inputs['anchor'])[0]
    graph['positive'] = inception.inception_v1(inputs['positive'], reuse=True)[0]
    graph['negative'] = inception.inception_v1(inputs['negative'], reuse=True)[0]

    # One graph trainer
    trainer = TripletTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory
                      )
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01)
                                        )
    trainer.train()
    shutil.rmtree(directory)
