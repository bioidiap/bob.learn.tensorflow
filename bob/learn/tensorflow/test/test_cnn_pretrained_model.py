#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
import bob.io.base
import os
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, TripletMemory, SiameseMemory, ScaleFactor
from bob.learn.tensorflow.loss import BaseLoss, TripletLoss, ContrastiveLoss
from bob.learn.tensorflow.trainers import Trainer, constant, TripletTrainer, SiameseTrainer
from bob.learn.tensorflow.utils import load_mnist
from bob.learn.tensorflow.layers import Conv2D, FullyConnected
from bob.learn.tensorflow.network import Embedding
from .test_cnn import dummy_experiment
from .test_cnn_scratch import validate_network


import tensorflow as tf
import shutil

"""
Some unit tests that create networks on the fly and load variables
"""

batch_size = 16
validation_batch_size = 400
iterations = 300
seed = 10


def scratch_network(input_pl):
    # Creating a random network
    slim = tf.contrib.slim

    with tf.device("/cpu:0"):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=10)

        scratch = slim.conv2d(input_pl, 16, [3, 3], activation_fn=tf.nn.relu,
                              stride=1,
                              weights_initializer=initializer,
                              scope='conv1')
        scratch = slim.max_pool2d(scratch, kernel_size=[2, 2], scope='pool1')
        scratch = slim.flatten(scratch, scope='flatten1')
        scratch = slim.fully_connected(scratch, 10,
                                       weights_initializer=initializer,
                                       activation_fn=None,
                                       scope='fc1')

    return scratch


def test_cnn_pretrained():
    # Preparing input data
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 28, 28, 1],
                                 batch_size=batch_size,
                                 data_augmentation=data_augmentation,
                                 normalizer=ScaleFactor())
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))
    directory = "./temp/cnn"

    # Creating a random network
    input_pl = train_data_shuffler("data", from_queue=True)
    graph = scratch_network(input_pl)
    embedding = Embedding(train_data_shuffler("data", from_queue=False), graph)

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.1, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.1))
    trainer.train()
    accuracy = validate_network(embedding, validation_data, validation_labels)

    assert accuracy > 80
    tf.reset_default_graph()

    del graph
    del loss
    del trainer
    del embedding
    # Training the network using a pre trained model
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean, name="loss")

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations*3,
                      analizer=None,
                      temp_dir=directory
                      )
    trainer.create_network_from_file(os.path.join(directory, "model.ckp"))
    trainer.train()
    embedding = Embedding(trainer.data_ph, trainer.graph)
    accuracy = validate_network(embedding, validation_data, validation_labels)
    assert accuracy > 90
    shutil.rmtree(directory)

    del loss
    del trainer


def test_triplet_cnn_pretrained():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = TripletMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size,
                                        data_augmentation=data_augmentation)
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    validation_data_shuffler = TripletMemory(validation_data, validation_labels,
                                             input_shape=[None, 28, 28, 1],
                                             batch_size=validation_batch_size)

    directory = "./temp/cnn"

    # Creating a random network
    input_pl = train_data_shuffler("data", from_queue=False)
    graph = dict()
    graph['anchor'] = scratch_network(input_pl['anchor'])
    graph['positive'] = scratch_network(input_pl['positive'])
    graph['negative'] = scratch_network(input_pl['negative'])

    # Loss for the softmax
    loss = TripletLoss(margin=4.)

    # One graph trainer
    trainer = TripletTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01))
    trainer.train()
    # Testing
    embedding = Embedding(trainer.data_ph['anchor'], trainer.graph['anchor'])
    eer = dummy_experiment(validation_data_shuffler, embedding)

    # The result is not so good
    assert eer < 0.25

    del graph
    del loss
    del trainer

    # Training the network using a pre trained model
    trainer = TripletTrainer(train_data_shuffler,
                             iterations=iterations*2,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_file(os.path.join(directory, "model.ckp"))
    trainer.train()

    embedding = Embedding(trainer.data_ph['anchor'], trainer.graph['anchor'])
    eer = dummy_experiment(validation_data_shuffler, embedding)

    # Now it is better
    assert eer < 0.20
    shutil.rmtree(directory)

    del trainer


def test_siamese_cnn_pretrained():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = SiameseMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size,
                                        data_augmentation=data_augmentation,
                                        normalizer=ScaleFactor())
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    validation_data_shuffler = SiameseMemory(validation_data, validation_labels,
                                             input_shape=[None, 28, 28, 1],
                                             batch_size=validation_batch_size,
                                             normalizer=ScaleFactor())
    directory = "./temp/cnn"

    # Creating graph
    input_pl = train_data_shuffler("data")
    graph = dict()
    graph['left'] = scratch_network(input_pl['left'])
    graph['right'] = scratch_network(input_pl['right'])

    # Loss for the softmax
    loss = ContrastiveLoss(contrastive_margin=4.)
    # One graph trainer
    trainer = SiameseTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01))
    trainer.train()

    # Testing
    #embedding = Embedding(train_data_shuffler("data", from_queue=False)['left'], graph['left'])
    embedding = Embedding(trainer.data_ph['left'], trainer.graph['left'])
    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.10

    del graph
    del loss
    del trainer

    trainer = SiameseTrainer(train_data_shuffler,
                             iterations=iterations*2,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_file(os.path.join(directory, "model.ckp"))
    trainer.train()

    #embedding = Embedding(train_data_shuffler("data", from_queue=False)['left'], trainer.graph['left'])
    embedding = Embedding(trainer.data_ph['left'], trainer.graph['left'])
    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.10

    shutil.rmtree(directory)

    del trainer
