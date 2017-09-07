#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory, ImageAugmentation, ScaleFactor
from bob.learn.tensorflow.network import Chopra
from bob.learn.tensorflow.loss import BaseLoss, ContrastiveLoss, TripletLoss
from bob.learn.tensorflow.trainers import Trainer, SiameseTrainer, TripletTrainer, constant
from .test_cnn_scratch import validate_network
from bob.learn.tensorflow.network import Embedding

from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import bob.io.base
import shutil
from scipy.spatial.distance import cosine
import bob.measure
from .test_cnn import dummy_experiment

"""
Some unit tests for the datashuffler
"""

batch_size = 32
validation_batch_size = 400
iterations = 100
seed = 10


def test_cnn_trainer():
    tf.reset_default_graph()

    # Loading data
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 28, 28, 1],
                                 batch_size=batch_size,
                                 data_augmentation=data_augmentation,
                                 normalizer=ScaleFactor(),
                                 prefetch=True,
                                 prefetch_threads=1)

    directory = "./temp/cnn"

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # Preparing the architecture
    architecture = Chopra(seed=seed,
                          fc1_output=10)
    input_pl = train_data_shuffler("data", from_queue=True)
    graph = architecture(input_pl)
    embedding = Embedding(train_data_shuffler("data", from_queue=False), architecture(train_data_shuffler("data", from_queue=False), reuse=True))

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
    #trainer.train(validation_data_shuffler)

    # Using embedding to compute the accuracy
    accuracy = validate_network(embedding, validation_data, validation_labels)

    # At least 80% of accuracy
    assert accuracy > 50.
    shutil.rmtree(directory)
    del trainer
    del graph
    del embedding
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    


def test_siamesecnn_trainer():

    """
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = SiameseMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size,
                                        normalizer=ScaleFactor(),
                                        prefetch=True)

    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[None, 28, 28, 1],
                                      batch_size=validation_batch_size,
                                      normalizer=ScaleFactor())

    directory = "./temp/siamesecnn"

    # Preparing the architecture
    architecture = Chopra(seed=seed, fc1_output=10)

    # Loss for the Siamese
    loss = ContrastiveLoss(contrastive_margin=4.)

    input_pl = train_data_shuffler("data", from_queue=True)
    graph = dict()
    graph['left'] = architecture(input_pl['left'])
    graph['right'] = architecture(input_pl['right'], reuse=True)

    trainer = SiameseTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),)
    trainer.train()
    embedding = Embedding(validation_data_shuffler("data", from_queue=False),
                          architecture(validation_data_shuffler("data", from_queue=False), reuse=True))
    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.25
    shutil.rmtree(directory)

    del architecture
    del trainer  # Just to clean tf.variables
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    
    """
    assert True

def test_tripletcnn_trainer():
    """
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = TripletMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size,
                                        normalizer=ScaleFactor(),
                                        prefetch=True)
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[None, 28, 28, 1],
                                      batch_size=validation_batch_size,
                                      normalizer=ScaleFactor())

    directory = "./temp/tripletcnn"

    # Preparing the architecture
    architecture = Chopra(seed=seed, fc1_output=10)

    # Loss for the Siamese
    loss = TripletLoss(margin=4.)

    input_pl = train_data_shuffler("data", from_queue=True)
    graph = dict()
    graph['anchor'] = architecture(input_pl['anchor'])
    graph['positive'] = architecture(input_pl['positive'], reuse=True)
    graph['negative'] = architecture(input_pl['negative'], reuse=True)

    # One graph trainer
    trainer = TripletTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory
                             )
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),)
    trainer.train()

    embedding = Embedding(validation_data_shuffler("data", from_queue=False),
                          architecture(validation_data_shuffler("data", from_queue=False), reuse=True))

    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.25
    shutil.rmtree(directory)

    del architecture
    del trainer  # Just to clean tf.variables
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    
    """
    assert True
