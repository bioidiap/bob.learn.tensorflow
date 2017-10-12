#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, scale_factor, TFRecord
from bob.learn.tensorflow.network import Embedding
from bob.learn.tensorflow.loss import mean_cross_entropy_loss, contrastive_loss, triplet_loss
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
iterations = 200
seed = 10
directory = "./temp/cnn_scratch"

slim = tf.contrib.slim


def scratch_network(train_data_shuffler, reuse=False):

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
    graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, reuse=reuse)

    return graph


def scratch_network_embeding_example(train_data_shuffler, reuse=False, get_embedding=False):

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
    graph = slim.fully_connected(graph, 30, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, reuse=reuse)

    if get_embedding:
        graph = tf.nn.l2_normalize(graph, dim=1, name="embedding")
    else:
        graph = slim.fully_connected(graph, 10, activation_fn=None, scope='logits',
                                     weights_initializer=initializer, reuse=reuse)
    
    return graph




def validate_network(embedding, validation_data, validation_labels, input_shape=[None, 28, 28, 1], normalizer=scale_factor):
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
    train_data_shuffler = Memory(train_data, train_labels,
                                 input_shape=[None, 28, 28, 1],
                                 batch_size=batch_size,
                                 normalizer=scale_factor)

    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    # Create scratch network
    logits = scratch_network(train_data_shuffler)
    labels = train_data_shuffler("label", from_queue=False)
    loss = mean_cross_entropy_loss(logits, labels)

    # Setting the placeholders
    embedding = Embedding(train_data_shuffler("data", from_queue=False), logits)

    # One graph trainer
    trainer = Trainer(train_data_shuffler,
                      iterations=iterations,
                      analizer=None,
                      temp_dir=directory)

    trainer.create_network_from_scratch(graph=logits,
                                        loss=loss,
                                        learning_rate=constant(0.01, name="regular_lr"),
                                        optimizer=tf.train.GradientDescentOptimizer(0.01),
                                        )

    trainer.train()
    accuracy = validate_network(embedding, validation_data, validation_labels)
    assert accuracy > 20
    shutil.rmtree(directory)
    del trainer
    tf.reset_default_graph() 
    assert len(tf.global_variables())==0


def test_cnn_tfrecord():
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
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=15, name="input")
    
    tfrecords_filename_val = "mnist_validation.tfrecords"
    create_tf_record(tfrecords_filename_val, validation_data, validation_labels)   
    filename_queue_val = tf.train.string_input_producer([tfrecords_filename_val], num_epochs=15, name="input_validation")

    # Creating the CNN using the TFRecord as input
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)

    validation_data_shuffler  = TFRecord(filename_queue=filename_queue_val,
                                         batch_size=2000)
                                         
    logits = scratch_network(train_data_shuffler)
    labels = train_data_shuffler("label", from_queue=False)

    validation_logits = scratch_network(validation_data_shuffler, reuse=True)
    validation_labels = validation_data_shuffler("label", from_queue=False)
    
    # Setting the placeholders
    # Loss for the softmax
    loss = mean_cross_entropy_loss(logits, labels)
    validation_loss = mean_cross_entropy_loss(validation_logits, validation_labels)

    # One graph trainer
    
    trainer = Trainer(train_data_shuffler,
                      validation_data_shuffler=validation_data_shuffler,
                      iterations=iterations, #It is supper fast
                      analizer=None,
                      temp_dir=directory)

    learning_rate = constant(0.01, name="regular_lr")

    trainer.create_network_from_scratch(graph=logits,
                                        validation_graph=validation_logits,
                                        loss=loss,
                                        validation_loss=validation_loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )
    
    trainer.train()
    os.remove(tfrecords_filename)
    os.remove(tfrecords_filename_val)    
    assert True
    tf.reset_default_graph()
    del trainer
    assert len(tf.global_variables())==0
    
    # Inference. TODO: Wrap this in a package
    file_name = os.path.join(directory, "model.ckp.meta")
    images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    graph = scratch_network(images, reuse=False)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(file_name, clear_devices=True)
    saver.restore(session, tf.train.latest_checkpoint(os.path.dirname("./temp/cnn_scratch/")))
    data = numpy.random.rand(2, 28, 28, 1).astype("float32")

    assert session.run(graph, feed_dict={images: data}).shape == (2, 10)

    tf.reset_default_graph()
    shutil.rmtree(directory)
    assert len(tf.global_variables())==0
    


def test_cnn_tfrecord_embedding_validation():
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
                                         
    logits = scratch_network_embeding_example(train_data_shuffler)
    labels = train_data_shuffler("label", from_queue=False)
    validation_logits = scratch_network_embeding_example(validation_data_shuffler, reuse=True, get_embedding=True)
    
    # Setting the placeholders
    # Loss for the softmax
    loss = mean_cross_entropy_loss(logits, labels)

    # One graph trainer
    
    trainer = Trainer(train_data_shuffler,
                      validation_data_shuffler=validation_data_shuffler,
                      validate_with_embeddings=True,
                      iterations=iterations, #It is supper fast
                      analizer=None,
                      temp_dir=directory)

    learning_rate = constant(0.01, name="regular_lr")

    trainer.create_network_from_scratch(graph=logits,
                                        validation_graph=validation_logits,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )
    
    trainer.train()
    os.remove(tfrecords_filename)
    os.remove(tfrecords_filename_val)    
    assert True
    tf.reset_default_graph()
    del trainer
    assert len(tf.global_variables())==0
    
    # Inference. TODO: Wrap this in a package
    file_name = os.path.join(directory, "model.ckp.meta")
    images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    graph = scratch_network_embeding_example(images, reuse=False)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(file_name, clear_devices=True)
    saver.restore(session, tf.train.latest_checkpoint(os.path.dirname("./temp/cnn_scratch/")))
    data = numpy.random.rand(2, 28, 28, 1).astype("float32")

    assert session.run(graph, feed_dict={images: data}).shape == (2, 10)

    tf.reset_default_graph()
    shutil.rmtree(directory)
    assert len(tf.global_variables())==0

