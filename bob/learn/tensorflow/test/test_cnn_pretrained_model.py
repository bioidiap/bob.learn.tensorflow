#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
import bob.io.base
import os
from bob.learn.tensorflow.datashuffler import Memory, TripletMemory, SiameseMemory, scale_factor
from bob.learn.tensorflow.loss import mean_cross_entropy_loss, contrastive_loss_deprecated, triplet_loss
from bob.learn.tensorflow.trainers import Trainer, constant, TripletTrainer, SiameseTrainer
from bob.learn.tensorflow.utils import load_mnist
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
iterations = 100
seed = 10


def scratch_network(input_pl, reuse=False):
    # Creating a random network
    slim = tf.contrib.slim

    with tf.device("/cpu:0"):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=10)

        scratch = slim.conv2d(input_pl, 16, [3, 3], activation_fn=tf.nn.relu,
                              stride=1,
                              weights_initializer=initializer,
                              scope='conv1', reuse=reuse)
        scratch = slim.max_pool2d(scratch, kernel_size=[2, 2], scope='pool1')
        scratch = slim.flatten(scratch, scope='flatten1')
        scratch = slim.fully_connected(scratch, 10,
                                       weights_initializer=initializer,
                                       activation_fn=None,
                                       scope='fc1', reuse=reuse)

    return scratch


def test_cnn_pretrained():
    tf.reset_default_graph()

    # Preparing input data
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers    
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 28, 28, 1],
                                 batch_size=batch_size)
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))
    directory = "./temp/cnn"

    # Creating a random network
    inputs = train_data_shuffler("data", from_queue=True)
    labels = train_data_shuffler("label", from_queue=True)
    logits = scratch_network(inputs)
    embedding = Embedding(train_data_shuffler("data", from_queue=False), logits)

    # Loss for the softmax
    loss = mean_cross_entropy_loss(logits, labels)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)
    trainer.create_network_from_scratch(graph=logits,
                                        loss=loss,
                                        learning_rate=constant(0.1, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.1))
    trainer.train()
    accuracy = validate_network(embedding, validation_data, validation_labels, normalizer=None)


    assert accuracy > 20
    tf.reset_default_graph()

    del logits
    del loss
    del trainer
    del embedding

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory
                      )
    trainer.create_network_from_file(os.path.join(directory, "model.ckp.meta"))
    trainer.train()
    embedding = Embedding(trainer.data_ph, trainer.graph)
    accuracy = validate_network(embedding, validation_data, validation_labels, normalizer=None)
    assert accuracy > 50
    shutil.rmtree(directory)

    del trainer
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    


def test_triplet_cnn_pretrained():
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = TripletMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size)
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    validation_data_shuffler = TripletMemory(validation_data, validation_labels,
                                             input_shape=[None, 28, 28, 1],
                                             batch_size=validation_batch_size)

    directory = "./temp/cnn"

    # Creating a random network
    inputs = train_data_shuffler("data", from_queue=False)
    graph = dict()
    graph['anchor'] = scratch_network(inputs['anchor'])
    graph['positive'] = scratch_network(inputs['positive'], reuse=True)
    graph['negative'] = scratch_network(inputs['negative'], reuse=True)

    # Loss for the softmax
    loss = triplet_loss(graph['anchor'], graph['positive'], graph['negative'], margin=4.)

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
    assert eer < 0.35

    del graph
    del loss
    del trainer

    # Training the network using a pre trained model
    trainer = TripletTrainer(train_data_shuffler,
                             iterations=iterations*2,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_file(os.path.join(directory, "model.ckp.meta"))
    trainer.train()

    embedding = Embedding(trainer.data_ph['anchor'], trainer.graph['anchor'])
    eer = dummy_experiment(validation_data_shuffler, embedding)

    # Now it is better
    assert eer < 0.30
    shutil.rmtree(directory)

    del trainer
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    


def test_siamese_cnn_pretrained():

    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = SiameseMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size)
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    validation_data_shuffler = SiameseMemory(validation_data, validation_labels,
                                             input_shape=[None, 28, 28, 1],
                                             batch_size=validation_batch_size)
    directory = "./temp/cnn"

    # Creating graph
    inputs = train_data_shuffler("data")
    labels = train_data_shuffler("label")
    graph = dict()
    graph['left'] = scratch_network(inputs['left'])
    graph['right'] = scratch_network(inputs['right'], reuse=True)

    # Loss for the softmax
    loss = contrastive_loss_deprecated(graph['left'], graph['right'], labels, contrastive_margin=4.)
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
    assert eer < 0.18

    del graph
    del loss
    del trainer

    trainer = SiameseTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_file(os.path.join(directory, "model.ckp.meta"))
    trainer.train()

    #embedding = Embedding(train_data_shuffler("data", from_queue=False)['left'], trainer.graph['left'])
    embedding = Embedding(trainer.data_ph['left'], trainer.graph['left'])
    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.18

    shutil.rmtree(directory)

    del trainer
    tf.reset_default_graph()
    assert len(tf.global_variables())==0

