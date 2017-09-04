#!/usr/bin/env python

import sys
import numpy as np
import random

from bob.learn.tensorflow.datashuffler import Memory, ScaleFactor
from bob.learn.tensorflow.network import MLP, Embedding
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_real_mnist, load_mnist
from bob.learn.tensorflow.utils.session import Session
from bob.learn.tensorflow.layers import rnn, rnn3d

import tensorflow as tf

import logging
logger = logging.getLogger("bob.learn.tf")

######################################################################


def test_network(embedding, test_data, test_labels):
    # Testing
    test_data_shuffler = Memory(test_data, test_labels,
                                input_shape=[None, 28*28],
                                batch_size=test_data.shape[0],
                                normalizer=ScaleFactor())

    [data, labels] = test_data_shuffler.get_batch()
    predictions = embedding(data)
    logger.info("Test prediction size {}".format(predictions.shape))
    acc = 100. * np.sum(np.argmax(predictions, axis=1) == labels) / predictions.shape[0]

    # gt = tf.placeholder(tf.int64, [None, ])
    # equal = tf.equal(tf.argmax(embedding.graph,1), gt)
    # accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    # ss = Session.instance().session
    # res = ss.run(embedding.graph, feed_dict={embedding.input: data})
    # res2 = ss.run(accuracy, feed_dict={embedding.input: data, gt: labels})
    # print("res {}".format(res.shape))
    # print("acc2 {}".format(res2))

    return acc


def test_lstm_trainer_on_mnist():
    """
    """
    batch_size = 64
    iterations = 300
    seed = 10
    learning_rate = 0.01

    n_input = 28 # MNIST data input (img shape: 28*28)
    n_steps = 27 # timesteps; 27 to avoid square images, easier to debug
    n_hidden = 128 # hidden layer num of features
    n_classes = 10 # MNIST total classes (0-9 digits)

    directory = "./temp/lstm"

    train_data, train_labels, test_data, test_labels = load_real_mnist(data_dir="mnist")

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 784],
                                 batch_size=batch_size,
                                 normalizer=ScaleFactor())

    # Preparing the architecture
    input_pl = train_data_shuffler("data", from_queue=False)

    version = "bob3d"

    # Original code using MLP
    if version == "mlp":
        architecture = MLP(10, hidden_layers=[20, 40])
        graph = architecture(input_pl)

    elif version == "lstm":
        slim = tf.contrib.slim
        # W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
        # b = tf.Variable(tf.random_normal([n_classes]))
        graph = input_pl[:, n_input:]
        graph = tf.reshape(graph, (-1, n_steps, n_input))
        graph = tf.unstack(graph, n_steps, 1)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = tf.nn.static_rnn(lstm_cell, graph, dtype=tf.float32)
        # graph = tf.matmul(outputs[-1], W) + b
        graph = outputs[-1]
        graph = slim.fully_connected(graph, n_classes, activation_fn=None)

    elif version == "bob":
        slim = tf.contrib.slim
        graph = input_pl[:, n_input:]
        graph = tf.reshape(graph, (-1, n_steps, n_input))
        graph = tf.unstack(graph, n_steps, 1)
        graph = rnn(graph, n_hidden)
        graph = slim.fully_connected(graph, n_classes, activation_fn=None)

    elif version == "bob3d":
        slim = tf.contrib.slim
        graph = input_pl[:, n_input:]
        graph = tf.reshape(graph, (-1, n_steps, n_input))
        graph = rnn3d(graph, n_hidden)
        graph = slim.fully_connected(graph, n_classes, activation_fn=None)

    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)

    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=constant(learning_rate, name="regular_lr"),
                                        optimizer=tf.train.AdamOptimizer(learning_rate))


    trainer.train()

    # Test
    embedding = Embedding(train_data_shuffler("data", from_queue=False), graph)
    accuracy = test_network(embedding, test_data, test_labels)
    logger.info("Accuracy {}".format(accuracy))
    assert accuracy > 0.8

def generate_data_at_time(t, n_steps, dt):
    """
    Generate a sequence for time t with n_steps previous times
    """
    dim = 3

    # The sequence t-n_steps*dt, ..., t-3*dt, t-2*dt, t-dt
    x = dt*(np.arange(n_steps, dtype=np.float64) + 1)
    x = -np.fliplr(x.reshape(1,-1))
    x += t

    # The values of the function before time t
    sequence_before = np.zeros((n_steps, dim))
    sequence_before[:,0] = np.sin(x)
    sequence_before[:,1] = np.cos(x)
    sequence_before[:,2] = np.cos(2*x) + np.sin(x)

    # The value to be predicted at time t
    target = np.zeros((1,dim))
    target[0,0] = np.sin(t)
    target[0,1] = np.cos(t)
    target[:,2] = np.cos(2*t) + np.sin(t)

    print("="*70)
    print(t)
    print(x)
    print(sequence_before)
    print(target)
    print("="*70)

    return t, x, sequence_before, target


def generate_training_data(n_train, n_steps, dt):
    """
    """
    t0, x0, s0, t0 = generate_data_at_time(0, n_steps, dt)
    dim = s0.shape[1]

    times     = np.zeros((n_train,1))
    sequences = np.zeros((n_train, n_steps, dim))
    targets   = np.zeros((n_train, dim))

    for n in range(n_train):
        t = 4*np.pi*(random.random() - 0.5)
        t0, x0, s0, y0 = generate_data_at_time(t, n_steps, dt)
        times[n,0]   = t0
        sequences[n] = s0
        targets[n]   = y0

    return times, sequences, targets


def test_lstm_trainer_on_real_functions():
    """
    """
    dt = 0.01
    n_train = 7
    n_steps = 3
    batch_size = 10

    # Generate train data in 3D matrix to use Memory class
    times, train_data, train_targets = generate_training_data(n_train, n_steps, dt)
    dim = train_data.shape[1]

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_targets,
                                 input_shape=[None, train_data[1], train_data[2]],
                                 batch_size=batch_size,
                                 normalizer=ScaleFactor())

    print(times)
    print(train_data)

    # Preparing the architecture
    input_pl = train_data_shuffler("data", from_queue=False)
    graph = rnn3d(graph, n_hidden)
    graph = slim.fully_connected(graph, dim, activation_fn=None)


    # # Creating datashufflers
    # train_data_shuffler = Memory(train_data, train_targets,
    #                              input_shape=[None, 784],
    #                              batch_size=batch_size,
    #                              normalizer=ScaleFactor())

test_lstm_trainer_on_mnist()

# test_lstm_trainer_on_real_functions()
