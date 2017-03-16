#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory, ImageAugmentation
from bob.learn.tensorflow.network import Chopra, SequenceNetwork
from bob.learn.tensorflow.loss import BaseLoss, ContrastiveLoss, TripletLoss
from bob.learn.tensorflow.trainers import Trainer, SiameseTrainer, TripletTrainer, constant
from .test_cnn_scratch import validate_network
import pkg_resources

from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import bob.io.base
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


def dummy_experiment(data_s, architecture):
    """
    Create a dummy experiment and return the EER
    """

    data_shuffler = object.__new__(Memory)
    data_shuffler.__dict__ = data_s.__dict__.copy()

    # Extracting features for enrollment
    enroll_data, enroll_labels = data_shuffler.get_batch()
    enroll_features = architecture(enroll_data)
    del enroll_data

    # Extracting features for probing
    probe_data, probe_labels = data_shuffler.get_batch()
    probe_features = architecture(probe_data)
    del probe_data

    # Creating models
    models = []
    for i in range(len(data_shuffler.possible_labels)):
        indexes_model = numpy.where(enroll_labels == data_shuffler.possible_labels[i])[0]
        models.append(numpy.mean(enroll_features[indexes_model, :], axis=0))

    # Probing
    positive_scores = numpy.zeros(shape=0)
    negative_scores = numpy.zeros(shape=0)

    for i in range(len(data_shuffler.possible_labels)):
        # Positive scoring
        indexes = probe_labels == data_shuffler.possible_labels[i]
        positive_data = probe_features[indexes, :]
        p = [cosine(models[i], positive_data[j]) for j in range(positive_data.shape[0])]
        positive_scores = numpy.hstack((positive_scores, p))

        # negative scoring
        indexes = probe_labels != data_shuffler.possible_labels[i]
        negative_data = probe_features[indexes, :]
        n = [cosine(models[i], negative_data[j]) for j in range(negative_data.shape[0])]
        negative_scores = numpy.hstack((negative_scores, n))

    threshold = bob.measure.eer_threshold((-1) * negative_scores, (-1) * positive_scores)
    far, frr = bob.measure.farfrr((-1) * negative_scores, (-1) * positive_scores, threshold)

    return (far + frr) / 2.


def test_cnn_trainer():

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    data_augmentation = ImageAugmentation()
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[28, 28, 1],
                                 batch_size=batch_size,
                                 data_augmentation=data_augmentation)

    directory = "./temp/cnn"

    # Preparing the architecture
    architecture = Chopra(seed=seed, fc1_output=10)

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(architecture=architecture,
                      loss=loss,
                      iterations=iterations,
                      analizer=None,
                      prefetch=False,
                      learning_rate=constant(0.05, name="regular_lr"),
                      optimizer=tf.train.AdamOptimizer(name="adam_softmax"),
                      temp_dir=directory
                      )

    trainer.train(train_data_shuffler)

    accuracy = validate_network(validation_data, validation_labels, architecture)

    # At least 80% of accuracy
    assert accuracy > 80.
    shutil.rmtree(directory)
    del trainer
    del architecture


def test_siamesecnn_trainer():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = SiameseMemory(train_data, train_labels,
                                        input_shape=[28, 28, 1],
                                        batch_size=batch_size)
    validation_data_shuffler = SiameseMemory(validation_data, validation_labels,
                                             input_shape=[28, 28, 1],
                                             batch_size=validation_batch_size)

    directory = "./temp/siamesecnn"

    # Preparing the architecture
    architecture = Chopra(seed=seed, fc1_output=10)

    # Loss for the Siamese
    loss = ContrastiveLoss(contrastive_margin=4.)

    # One graph trainer
    trainer = SiameseTrainer(architecture=architecture,
                             loss=loss,
                             iterations=iterations,
                             prefetch=False,
                             analizer=None,
                             learning_rate=constant(0.05, name="siamese_lr"),
                             optimizer=tf.train.AdamOptimizer(name="adam_siamese"),
                             temp_dir=directory
                             )

    trainer.train(train_data_shuffler)

    eer = dummy_experiment(validation_data_shuffler, architecture)

    # At least 80% of accuracy
    assert eer < 0.25
    shutil.rmtree(directory)

    del architecture
    del trainer  # Just to clean tf.variables


def test_tripletcnn_trainer():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = TripletMemory(train_data, train_labels,
                                        input_shape=[28, 28, 1],
                                        batch_size=batch_size)
    validation_data_shuffler = TripletMemory(validation_data, validation_labels,
                                             input_shape=[28, 28, 1],
                                             batch_size=validation_batch_size)

    directory = "./temp/tripletcnn"

    # Preparing the architecture
    architecture = Chopra(seed=seed, fc1_output=10)

    # Loss for the Siamese
    loss = TripletLoss(margin=4.)

    # One graph trainer
    trainer = TripletTrainer(architecture=architecture,
                             loss=loss,
                             iterations=iterations,
                             prefetch=False,
                             analizer=None,
                             learning_rate=constant(0.05, name="triplet_lr"),
                             optimizer=tf.train.AdamOptimizer(name="adam_triplet"),
                             temp_dir=directory
                             )

    trainer.train(train_data_shuffler)

    # Testing
    eer = dummy_experiment(validation_data_shuffler, architecture)

    # At least 80% of accuracy
    assert eer < 0.25
    shutil.rmtree(directory)

    del architecture
    del trainer  # Just to clean tf.variables
