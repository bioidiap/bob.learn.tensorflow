#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import TFRecord
from bob.learn.tensorflow.loss import mean_cross_entropy_loss, mean_cross_entropy_center_loss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
from bob.learn.tensorflow.network.utils import append_logits
import tensorflow as tf
import shutil
import os

"""
Some unit tests that create networks on the fly
"""

batch_size = 16
validation_batch_size = 400
iterations = 200
seed = 10
directory = "./temp/cnn_scratch"

slim = tf.contrib.slim


def scratch_network_embeding_example(train_data_shuffler, reuse=False):

    if isinstance(train_data_shuffler, tf.Tensor):
        inputs = train_data_shuffler
    else:
        inputs = train_data_shuffler("data", from_queue=False)

    # Creating a random network
    initializer = tf.contrib.layers.xavier_initializer(seed=seed)
    graph = slim.conv2d(inputs, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                        weights_initializer=initializer, reuse=reuse)
    graph = slim.max_pool2d(graph, [4, 4], scope='pool1')
    graph = slim.flatten(graph, scope='flatten1')
    prelogits = slim.fully_connected(graph, 30, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, reuse=reuse)

    return prelogits

def test_center_loss_tfrecord_embedding_validation():
    tf.reset_default_graph()

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = train_data.astype("float32") *  0.00390625
    validation_data = validation_data.astype("float32") *  0.00390625    

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def create_tf_record(tfrecords_filename, data, labels):
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        #for i in range(train_data.shape[0]):
        for i in range(6000):
            img = data[i]
            img_raw = img.tostring()
            
            feature = {'train/data': _bytes_feature(img_raw),
                       'train/label': _int64_feature(labels[i])
                      }
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()
    
    tf.reset_default_graph()
    
    # Creating the tf record
    tfrecords_filename = "mnist_train.tfrecords"    
    create_tf_record(tfrecords_filename, train_data, train_labels)
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=55, name="input")
    
    tfrecords_filename_val = "mnist_validation.tfrecords"
    create_tf_record(tfrecords_filename_val, validation_data, validation_labels)   
    filename_queue_val = tf.train.string_input_producer([tfrecords_filename_val], num_epochs=55, name="input_validation")


    # Creating the CNN using the TFRecord as input
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)

    validation_data_shuffler  = TFRecord(filename_queue=filename_queue_val,
                                         batch_size=2000)
                                         
    prelogits = scratch_network_embeding_example(train_data_shuffler)
    logits = append_logits(prelogits, n_classes=10)
    validation_graph = tf.nn.l2_normalize(scratch_network_embeding_example(validation_data_shuffler, reuse=True), 1)

    labels = train_data_shuffler("label", from_queue=False)
    
    # Setting the placeholders
    # Loss for the softmax
    loss =  mean_cross_entropy_center_loss(logits, prelogits, labels, n_classes=10, factor=0.1)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      validation_data_shuffler=validation_data_shuffler,
                      validate_with_embeddings=True,
                      iterations=iterations, #It is supper fast
                      analizer=None,
                      temp_dir=directory)

    learning_rate = constant(0.01, name="regular_lr")

    trainer.create_network_from_scratch(graph=logits,
                                        validation_graph=validation_graph,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        prelogits=prelogits
                                        )
    trainer.train()
    
    assert True
    tf.reset_default_graph()
    del trainer
    assert len(tf.global_variables())==0

    del train_data_shuffler
    del validation_data_shuffler

    ##### 2 Continuing the training
    
    # Creating the CNN using the TFRecord as input
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)

    validation_data_shuffler  = TFRecord(filename_queue=filename_queue_val,
                                         batch_size=2000)
    
    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      validation_data_shuffler=validation_data_shuffler,
                      validate_with_embeddings=True,
                      iterations=2, #It is supper fast
                      analizer=None,
                      temp_dir=directory)

    trainer.create_network_from_file(directory)
    trainer.train()
    
    os.remove(tfrecords_filename)
    os.remove(tfrecords_filename_val)    

    tf.reset_default_graph()
    shutil.rmtree(directory)
    assert len(tf.global_variables())==0


