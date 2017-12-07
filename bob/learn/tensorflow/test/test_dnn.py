#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, scale_factor
from bob.learn.tensorflow.network import mlp, Embedding
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
import tensorflow as tf
import shutil

"""
Some unit tests for the datashuffler
"""

batch_size = 16
validation_batch_size = 400
iterations = 200
seed = 10


def validate_network(embedding, validation_data, validation_labels):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=[None, 28*28],
                                      batch_size=validation_batch_size)

    [data, labels] = validation_data_shuffler.get_batch()
    predictions = embedding(data)
    accuracy = 100. * numpy.sum(numpy.argmax(predictions, axis=1) == labels) / predictions.shape[0]

    return accuracy


def test_dnn_trainer():
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 784],
                                 batch_size=batch_size)

    directory = "./temp/dnn"

    # Preparing the architecture
    

    inputs = train_data_shuffler("data", from_queue=False)
    labels = train_data_shuffler("label", from_queue=False)
    logits = mlp(inputs, 10, hidden_layers=[20, 40])

    # Loss for the softmax
    loss = mean_cross_entropy_loss(logits, labels)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory
                      )

    trainer.create_network_from_scratch(graph=logits,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),
                                        )

    trainer.train()
    embedding = Embedding(train_data_shuffler("data", from_queue=False), logits)
    accuracy = validate_network(embedding, validation_data, validation_labels)

    # At least 50% of accuracy for the DNN
    assert accuracy > 50.
    shutil.rmtree(directory)

    del trainer  # Just to clean the variables
    tf.reset_default_graph()
    assert len(tf.global_variables())==0    
