#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, ImageAugmentation, ScaleFactor, Linear, TFRecord
from bob.learn.tensorflow.network import Embedding
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import shutil
import os

"""
Some unit tests that create networks on the fly
"""

batch_size = 16
validation_batch_size = 400
iterations = 300
seed = 10
directory = "./temp/cnn_scratch"

slim = tf.contrib.slim


def scratch_network(train_data_shuffler, reuse=False):

    inputs = train_data_shuffler("data", from_queue=False)

    # Creating a random network
    initializer = tf.contrib.layers.xavier_initializer(seed=seed)
    graph = slim.conv2d(inputs, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                        weights_initializer=initializer, reuse=reuse)
    graph = slim.max_pool2d(graph, [4, 4], scope='pool1')
    graph = slim.flatten(graph, scope='flatten1')
    graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, reuse=reuse)

    return graph


def validate_network(embedding, validation_data, validation_labels, input_shape=[None, 28, 28, 1], normalizer=ScaleFactor()):
    # Testing
    validation_data_shuffler = Memory(validation_data, validation_labels,
                                      input_shape=input_shape,
                                      batch_size=validation_batch_size,
                                      normalizer=normalizer)

    [data, labels] = validation_data_shuffler.get_batch()
    predictions = embedding(data)
    accuracy = 100. * numpy.sum(numpy.argmax(predictions, axis=1) == labels) / predictions.shape[0]

    return accuracy


def test_cnn_trainer_scratch():
    tf.reset_default_graph()

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
    # Create scratch network
    graph = scratch_network(train_data_shuffler)

    # Setting the placeholders
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
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),
                                        )

    trainer.train()
    accuracy = validate_network(embedding, validation_data, validation_labels)
    assert accuracy > 70
    shutil.rmtree(directory)
    del trainer
    
    
    
    
def test_cnn_trainer_scratch_tfrecord():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = train_data.astype("float32") *  0.00390625

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def create_tf_record(tfrecords_filename):
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        for i in range(train_data.shape[0]):
            img = train_data[i]
            img_raw = img.tostring()
            
            feature = {'train/image': _bytes_feature(img_raw),
                       'train/label': _int64_feature(train_labels[i])
                      }
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    tf.reset_default_graph()
    
    # Creating the tf record
    tfrecords_filename = "mnist_train.tfrecords"
    create_tf_record(tfrecords_filename)   
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1, name="input")

    # Creating the CNN using the TFRecord as input
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)
    graph = scratch_network(train_data_shuffler)

    # Setting the placeholders
    # Loss for the softmax
    loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)

    learning_rate = constant(0.01, name="regular_lr")
    trainer.create_network_from_scratch(graph=graph,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )

    trainer.train()
    os.remove(tfrecords_filename)
    assert True


