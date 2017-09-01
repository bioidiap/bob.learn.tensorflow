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
    tf.reset_default_graph()

    #import ipdb; ipdb.set_trace();

    #train_data, train_labels, validation_data, validation_labels = load_mnist()
    #train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    tfrecords_filename = "/idiap/user/tpereira/gitlab/workspace_HTFace/mnist_train.tfrecords"
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1, name="XUXA")
    train_data_shuffler  = TFRecord(filename_queue=filename_queue,
                                    batch_size=batch_size)

    # Creating datashufflers
    # Create scratch network
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
    #accuracy = validate_network(embedding, validation_data, validation_labels)
    #assert accuracy > 70
    #shutil.rmtree(directory)
    #del trainer    
    
    
    
def test_xuxa():
    tfrecords_filename = '/idiap/user/tpereira/gitlab/workspace_HTFace/mnist_train.tfrecords'
    def read_and_decode(filename_queue):

        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        
        _, serialized_example = reader.read(filename_queue)
        
        
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['train/image'], tf.float32)
        
        # Cast label data into int32
        label = tf.cast(features['train/label'], tf.int64)
        
        # Reshape image data into the original shape
        image = tf.reshape(image, [28, 28, 1])
        
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=32, capacity=1000, num_threads=1, min_after_dequeue=1, name="XUXA1")

        return images, labels



    slim = tf.contrib.slim


    def scratch_network(inputs, reuse=False):

        # Creating a random network
        initializer = tf.contrib.layers.xavier_initializer(seed=10)
        graph = slim.conv2d(inputs, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1',
                            weights_initializer=initializer, reuse=reuse)
        graph = slim.max_pool2d(graph, [4, 4], scope='pool1')
        graph = slim.flatten(graph, scope='flatten1')
        graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1',
                                     weights_initializer=initializer, reuse=reuse)

        return graph

    def create_general_summary(predictor):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """

        # Train summary
        tf.summary.scalar('loss', predictor)
        return tf.summary.merge_all()


    #create_tf_record()


    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=5, name="XUXA")

    images, labels = read_and_decode(filename_queue)
    graph = scratch_network(images)
    predictor = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=graph, labels=labels)
    loss = tf.reduce_mean(predictor)

    global_step = tf.contrib.framework.get_or_create_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, global_step=global_step)



    print("Batching")
    #import ipdb; ipdb.set_trace()
    sess = tf.Session()
    #with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    saver = tf.train.Saver(var_list=tf.global_variables() + tf.local_variables())

    train_summary_writter = tf.summary.FileWriter('./tf-record/train', sess.graph)
    summary_op = create_general_summary(loss)

        
    #tf.global_variables_initializer().run(session=self.session)

    # Any preprocessing here ...

    ############# Batching ############

    # Creates batches by randomly shuffling tensors
    #images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    #images, labels = tf.train.batch([image, label], batch_size=10)


    #import ipdb; ipdb.set_trace();
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)
    #sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    #import ipdb; ipdb.set_trace();

    #import ipdb; ipdb.set_trace()
    for i in range(10):
        _, l, summary = sess.run([optimizer, loss, summary_op])
        print l

        #img, lbl = sess.run([images, labels])        
        #print img.shape
        #print lbl
        train_summary_writter.add_summary(summary, i)


    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)    
    x = 0
    train_summary_writter.close()
    saver.save(sess, "xuxa.ckp")






