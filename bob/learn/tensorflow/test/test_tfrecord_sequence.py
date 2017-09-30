#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 13 Sep 2017 13:22 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, TFRecordSequence
from bob.learn.tensorflow.network import Embedding
from bob.learn.tensorflow.loss import MeanSoftMaxLoss
from bob.learn.tensorflow.trainers import Trainer, constant, exponential_decay
from bob.learn.tensorflow.utils import load_mnist
from bob.learn.tensorflow.layers import lstm
import tensorflow as tf
import shutil
import os

import logging
logger = logging.getLogger("bob.learn")

slim = tf.contrib.slim


def scratch_lstm_network(train_data_shuffler, lstm_cell_size=64, batch_size=10,
                         num_time_steps=28, num_classes=10, seed=10, reuse=False):
    inputs = train_data_shuffler("data", from_queue=False)

    initializer = tf.contrib.layers.xavier_initializer(seed=seed)

    # Creating an LSTM network
    graph = lstm(inputs, lstm_cell_size, num_time_steps=num_time_steps, batch_size=batch_size,
                 output_activation_size=num_classes, scope='lstm',
                 weights_initializer=initializer, activation=tf.nn.relu, reuse=reuse)

    # fully connect the LSTM output to the classes
    graph = slim.fully_connected(graph, num_classes, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, reuse=reuse)

    return graph


def validate_network(embedding, validation_data, validation_labels,
                     input_shape=[None, 28, 28, 1], validation_batch_size=10,
                     normalizer=ScaleFactor()):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=input_shape,
                                      batch_size=validation_batch_size,
                                      normalizer=normalizer)

    valid_range = 10
    accuracy = 0
    for i in range(valid_range):
        [data, labels] = validation_data_shuffler.get_batch()
        predictions = embedding(data)
        accuracy += 100. * numpy.sum(numpy.argmax(predictions, axis=1) == labels) / predictions.shape[0]
    accuracy /= valid_range
    logger.info("Validation accuracy = {0}".format(accuracy))
    return accuracy


def create_tf_record(tfrecords_filename, train=True):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # tf.reset_default_graph()

    data, labels, validation_data, validation_labels = load_mnist()
    data = data.astype("float32") * 0.00390625
    validation_data = validation_data.astype("float32") * 0.00390625

    if not train:
        data = validation_data
        labels = validation_labels

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    # for i in range(train_data.shape[0]):
    for i in range(6000):
        img = data[i]
        img_raw = img.tostring()

        feature = {'train/data': _bytes_feature(img_raw),
                   'train/label': _int64_feature(labels[i])
                   }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def test_lstm_trainer():
    # define constants that describe data and define experiments
    lstm_cell_size = 64
    num_time_steps = 28
    feature_size = 28
    batch_size = 16
    input_shape = [None, num_time_steps, feature_size, 1]
    iterations = 500
    seed = 10

    num_classes = 10
    directory = "./temp/lstm_scratch"

    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], ) + tuple(input_shape[1:]))

    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=input_shape,
                                 batch_size=batch_size,
                                 data_augmentation=None,
                                 normalizer=ScaleFactor())

    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], ) + tuple(input_shape[1:]))
    # Create scratch network
    graph = scratch_lstm_network(train_data_shuffler,
                                 lstm_cell_size=lstm_cell_size,
                                 batch_size=batch_size,
                                 num_time_steps=num_time_steps,
                                 seed=seed,
                                 num_classes=num_classes)

    # Setting the placeholders
    embedding = Embedding(train_data_shuffler("data", from_queue=False), tf.nn.softmax(graph), normalizer=None)

    # Loss for the softmax
    loss = MeanSoftMaxLoss()

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)

    learning_rate = constant(0.01, name="regular_lr")
    # learning_rate = exponential_decay(base_learning_rate=0.01, decay_steps=500, weight_decay=0.99, name='exp_lr')
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.AdamOptimizer(learning_rate),
                                        )

    trainer.train()
    accuracy = validate_network(embedding, validation_data, validation_labels, validation_batch_size=batch_size)
    logger.info("Ran for {0} full epochs".format(train_data_shuffler.epoch))

    assert accuracy > 30
    shutil.rmtree(directory)
    del trainer
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


def test_lstm_tfrecord():
    # define constants that describe data and define experiments
    lstm_cell_size = 64
    num_time_steps = 28
    feature_size = 28
    sliding_win_len = 26  # we feed 26 values into LSTM
    sliding_win_step = 2  # two sliding windows our of 28 values
    batch_size = 5
    validation_batch_size = 1
    input_shape = [None, num_time_steps, feature_size, 1]
    iterations = 100
    seed = 10
    directory = "./temp/lstm_scratch"

    num_classes = 10

    # Creating the tf record
    tfrecords_filename = "mnist_train.tfrecords"
    create_tf_record(tfrecords_filename)
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=15, name="input")

    tfrecords_filename_val = "mnist_validation.tfrecords"
    create_tf_record(tfrecords_filename_val, train=False)
    filename_queue_val = tf.train.string_input_producer([tfrecords_filename_val], num_epochs=15,
                                                        name="input_validation")

    # Creating TFRecord
    train_data_shuffler = TFRecordSequence(filename_queue=filename_queue,
                                           input_shape=input_shape,
                                           batch_size=batch_size,
                                           prefetch_threads=1,
                                           prefetch_capacity=50,
                                           sliding_win_len=sliding_win_len,
                                           sliding_win_step=sliding_win_step,
                                           min_after_dequeue=1)

    validation_data_shuffler = TFRecordSequence(filename_queue=filename_queue_val,
                                                input_shape=input_shape,
                                                batch_size=validation_batch_size,
                                                prefetch_threads=1,
                                                prefetch_capacity=50,
                                                sliding_win_len=sliding_win_len,
                                                sliding_win_step=sliding_win_step,
                                                min_after_dequeue=1)

    num_sliding_wins = (num_time_steps - sliding_win_len) // sliding_win_step + 1
    # after we generate sliding windows, the num_time_steps is the same as sliding windwos length
    num_time_steps = sliding_win_len
    graph = scratch_lstm_network(train_data_shuffler,
                                 lstm_cell_size=lstm_cell_size,
                                 batch_size=num_sliding_wins * batch_size,
                                 num_time_steps=num_time_steps,
                                 seed=seed,
                                 num_classes=num_classes)

    validation_graph = scratch_lstm_network(validation_data_shuffler,
                                            lstm_cell_size=lstm_cell_size,
                                            batch_size=num_sliding_wins * validation_batch_size,
                                            num_time_steps=num_time_steps,
                                            seed=seed,
                                            num_classes=num_classes,
                                            reuse=True)

    # Loss for the softmax
    loss = MeanSoftMaxLoss()

    # One graph trainer

    trainer = Trainer(train_data_shuffler,
                      validation_data_shuffler=None,
                      iterations=iterations,  # It is supper fast
                      analizer=None,
                      temp_dir=directory)

    learning_rate = constant(0.01, name="regular_lr")

    trainer.create_network_from_scratch(graph=graph,
                                        validation_graph=validation_graph,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.AdamOptimizer(learning_rate),
                                        )

    trainer.train()
    os.remove(tfrecords_filename)
    os.remove(tfrecords_filename_val)
    assert True
    tf.reset_default_graph()
    shutil.rmtree(directory)
    del trainer
    assert len(tf.global_variables()) == 0
