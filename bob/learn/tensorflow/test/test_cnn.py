#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory, scale_factor
from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.loss import mean_cross_entropy_loss, contrastive_loss_deprecated, triplet_loss_deprecated
from bob.learn.tensorflow.trainers import Trainer, SiameseTrainer, TripletTrainer, constant
from bob.learn.tensorflow.test.test_cnn_scratch import validate_network
from bob.learn.tensorflow.network import Embedding, light_cnn9
from bob.learn.tensorflow.network.utils import append_logits


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
iterations = 200
seed = 10
numpy.random.seed(seed)

def dummy_experiment(data_s, embedding):
    """
    Create a dummy experiment and return the EER
    """
    data_shuffler = object.__new__(Memory)
    data_shuffler.__dict__ = data_s.__dict__.copy()

    # Extracting features for enrollment
    enroll_data, enroll_labels = data_shuffler.get_batch()
    enroll_features = embedding(enroll_data)
    del enroll_data

    # Extracting features for probing
    probe_data, probe_labels = data_shuffler.get_batch()
    probe_features = embedding(probe_data)
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
    tf.reset_default_graph()

    # Loading data
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    # * 0.00390625
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 28, 28, 1],
                                 batch_size=batch_size)

    directory = "./temp/cnn"

    # Preparing the graph
    inputs = train_data_shuffler("data", from_queue=True)
    labels = train_data_shuffler("label", from_queue=True)
    logits = append_logits(dummy(inputs)[0], n_classes=10)
    
    # Loss for the softmax
    loss = mean_cross_entropy_loss(logits, labels)
    
    embedding = Embedding(inputs, logits)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory
                      )
    learning_rate=constant(0.1, name="regular_lr")
    trainer.create_network_from_scratch(graph=logits,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )
    trainer.train()
    #trainer.train(validation_data_shuffler)

    # Using embedding to compute the accuracy    
    accuracy = validate_network(embedding, validation_data, validation_labels, normalizer=None)

    # At least 20% of accuracy
    assert accuracy > 20.
    shutil.rmtree(directory)
    del trainer
    del logits
    tf.reset_default_graph()
    assert len(tf.global_variables())==0


def test_lightcnn_trainer():
    tf.reset_default_graph()

    # generating fake data
    train_data = numpy.random.normal(0, 0.2, size=(100, 128, 128, 1))
    train_data = numpy.vstack((train_data, numpy.random.normal(2, 0.2, size=(100, 128, 128, 1))))
    train_labels = numpy.hstack((numpy.zeros(100), numpy.ones(100))).astype("uint64")
    
    validation_data = numpy.random.normal(0, 0.2, size=(100, 128, 128, 1))
    validation_data = numpy.vstack((validation_data, numpy.random.normal(2, 0.2, size=(100, 128, 128, 1))))
    validation_labels = numpy.hstack((numpy.zeros(100), numpy.ones(100))).astype("uint64")

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 128, 128, 1],
                                 batch_size=batch_size,
                                 normalizer=scale_factor)

    directory = "./temp/cnn"

    # Preparing the architecture
    inputs = train_data_shuffler("data", from_queue=True)
    labels = train_data_shuffler("label", from_queue=True)
    prelogits = light_cnn9(inputs)[0]
    logits = append_logits(prelogits, n_classes=10)
    
    embedding = Embedding(train_data_shuffler("data", from_queue=False), logits)
    
    # Loss for the softmax
    loss = mean_cross_entropy_loss(logits, labels)


    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=4,
                      analizer=None,
                      temp_dir=directory
                      )
    trainer.create_network_from_scratch(graph=logits,
                                        loss=loss,
                                        learning_rate=constant(0.001, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.001),
                                        )
    trainer.train()
    #trainer.train(validation_data_shuffler)

    # Using embedding to compute the accuracy
    accuracy = validate_network(embedding, validation_data, validation_labels, input_shape=[None, 128, 128, 1], normalizer=scale_factor)
    assert True
    shutil.rmtree(directory)
    del trainer
    del logits
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    


def test_siamesecnn_trainer():
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = SiameseMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size)
    validation_data_shuffler = SiameseMemory(validation_data, validation_labels,
                                             input_shape=[None, 28, 28, 1],
                                             batch_size=validation_batch_size)
    directory = "./temp/siamesecnn"

    # Building the graph
    inputs = train_data_shuffler("data")
    labels = train_data_shuffler("label")
    graph = dict()
    graph['left'] = dummy(inputs['left'])[0]
    graph['right'] = dummy(inputs['right'], reuse=True)[0]

    # Loss for the Siamese
    loss = contrastive_loss_deprecated(graph['left'], graph['right'], labels, contrastive_margin=4.)

    trainer = SiameseTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory)

    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),)
                                        
    trainer.train()
    embedding = Embedding(train_data_shuffler("data", from_queue=False)['left'], graph['left'])
    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.15
    shutil.rmtree(directory)

    del trainer  # Just to clean tf.variables
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    


def test_tripletcnn_trainer():
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Creating datashufflers
    train_data_shuffler = TripletMemory(train_data, train_labels,
                                        input_shape=[None, 28, 28, 1],
                                        batch_size=batch_size)
    validation_data_shuffler = TripletMemory(validation_data, validation_labels,
                                             input_shape=[None, 28, 28, 1],
                                             batch_size=validation_batch_size)

    directory = "./temp/tripletcnn"

    inputs = train_data_shuffler("data")
    labels = train_data_shuffler("label")
    graph = dict()
    graph['anchor'] = dummy(inputs['anchor'])[0]
    graph['positive'] = dummy(inputs['positive'], reuse=True)[0]
    graph['negative'] = dummy(inputs['negative'], reuse=True)[0]

    loss = triplet_loss_deprecated(graph['anchor'], graph['positive'], graph['negative'])

    # One graph trainer
    trainer = TripletTrainer(train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=directory
                             )
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(0.1, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.1),)
    trainer.train()
    embedding = Embedding(train_data_shuffler("data", from_queue=False)['anchor'], graph['anchor'])
    eer = dummy_experiment(validation_data_shuffler, embedding)
    assert eer < 0.25
    shutil.rmtree(directory)

    del trainer  # Just to clean tf.variables
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    

