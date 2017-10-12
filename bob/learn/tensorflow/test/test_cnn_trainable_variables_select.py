#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import numpy
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf
import os
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
from bob.learn.tensorflow.datashuffler import TFRecord
from bob.learn.tensorflow.trainers import Trainer, constant


batch_size = 16
validation_batch_size = 400
iterations = 200
seed = 10

directory = "./temp/trainable_variables/"
step1_path = os.path.join(directory, "step1")
step2_path = os.path.join(directory, "step2")

slim = tf.contrib.slim


def base_network(train_data_shuffler, reuse=False, get_embedding=False, trainable=True):

    if isinstance(train_data_shuffler, tf.Tensor):
        inputs = train_data_shuffler
    else:
        inputs = train_data_shuffler("data", from_queue=False)

    # Creating a random network
    initializer = tf.contrib.layers.xavier_initializer(seed=seed)
    graph = slim.conv2d(inputs, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                        weights_initializer=initializer, reuse=reuse, trainable=trainable)
    graph = slim.max_pool2d(graph, [4, 4], scope='pool1')
    graph = slim.flatten(graph, scope='flatten1')
    graph = slim.fully_connected(graph, 30, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, reuse=reuse, trainable=trainable)

    if get_embedding:
        graph = graph
    else:
        graph = slim.fully_connected(graph, 10, activation_fn=None, scope='logits',
                                     weights_initializer=initializer, reuse=reuse)
    
    return graph


def amendment_network(graph, reuse=False, get_embedding=False):

    initializer = tf.contrib.layers.xavier_initializer(seed=seed)
    graph = slim.fully_connected(graph, 30, activation_fn=None, scope='fc2',
                                 weights_initializer=initializer, reuse=reuse)

    graph = slim.fully_connected(graph, 30, activation_fn=None, scope='fc3',
                                 weights_initializer=initializer, reuse=reuse)

    if get_embedding:
        graph = tf.nn.l2_normalize(graph, dim=1, name="embedding")
    else:
        graph = slim.fully_connected(graph, 10, activation_fn=None, scope='logits',
                                     weights_initializer=initializer, reuse=reuse)
    
    return graph

def test_trainable_variables():

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
                       'train/label': _int64_feature(labels[i])}
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


    ######## 1 - BASE NETWORK #########    


    tfrecords_filename = "mnist_train.tfrecords"    
    create_tf_record(tfrecords_filename, train_data, train_labels)
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1, name="input")

    # Doing the first training
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)
    
    logits = base_network(train_data_shuffler)
    labels = train_data_shuffler("label", from_queue=True)
    loss = mean_cross_entropy_loss(logits, labels)

    trainer = Trainer(train_data_shuffler,
                  iterations=iterations, #It is supper fast
                  analizer=None,
                  temp_dir=step1_path)

    learning_rate = constant(0.01, name="regular_lr")
    trainer.create_network_from_scratch(graph=logits,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )
    trainer.train()
    
    # Saving the cov after first training
    conv1_after_first_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1')[0].eval(session=trainer.session)[0]

    del trainer
    del filename_queue
    del train_data_shuffler
    tf.reset_default_graph()
        
    ######## 2 - AMEMDING NETWORK ######### 

    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1, name="input")
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)
    
    # Here I'm creating the base network not trainable
    embedding = base_network(train_data_shuffler, get_embedding=True, trainable=False)
    embedding = amendment_network(embedding)
    labels = train_data_shuffler("label", from_queue=True)
    loss = mean_cross_entropy_loss(embedding, labels)

    trainer = Trainer(train_data_shuffler,
                  iterations=iterations, #It is supper fast
                  analizer=None,
                  temp_dir=step2_path)

    learning_rate = constant(0.01, name="regular_lr")
    trainer.create_network_from_scratch(graph=embedding,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                        )

    # Loading two layers from the "old" model
    trainer.load_variables_from_external_model(step1_path, var_list=['conv1', 'fc1'])

    conv1_restored = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1')[0].eval(session=trainer.session)[0]
  
    assert numpy.allclose(conv1_after_first_train, conv1_restored)  

    # Second round of training    
    trainer.train()    
    conv1_after_second_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1')[0].eval(session=trainer.session)[0]

    # Since conv1 was set as NON TRAINABLE, both have to match
    assert numpy.allclose(conv1_after_first_train, conv1_after_second_train)  

