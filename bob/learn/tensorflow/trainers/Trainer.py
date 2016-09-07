#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
from ..network import SequenceNetwork
import threading
import numpy
import os
import bob.io.base

class Trainer(object):

    def __init__(self,

                 architecture=None,
                 use_gpu=False,
                 loss=None,
                 temp_dir="",

                 ###### training options ##########
                 convergence_threshold = 0.01,
                 iterations=5000,
                 base_lr=0.001,
                 momentum=0.9,
                 weight_decay=0.95,

                 # The learning rate policy
                 snapshot=100):

        self.loss = loss
        self.loss_instance = None
        self.optimizer = None
        self.temp_dir=temp_dir


        self.architecture = architecture
        self.use_gpu = use_gpu

        self.iterations = iterations
        self.snapshot = snapshot
        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.convergence_threshold = convergence_threshold

    def train(self, data_shuffler):
        """
        Do the loop forward --> backward --|
                      ^--------------------|
        """

        def start_thread():
            threads = []
            for n in range(1):
                t = threading.Thread(target=load_and_enqueue)
                t.daemon = True  # thread will close when parent quits
                t.start()
                threads.append(t)
            return threads

        def load_and_enqueue():
            """
            Injecting data in the place holder queue
            """

            #while not thread_pool.should_stop():
            for i in range(self.iterations):
                train_data, train_labels = data_shuffler.get_batch()

                feed_dict = {train_placeholder_data: train_data,
                             train_placeholder_labels: train_labels}

                session.run(enqueue_op, feed_dict=feed_dict)

        # Defining place holders
        train_placeholder_data, train_placeholder_labels = data_shuffler.get_placeholders_forprefetch(name="train")
        validation_placeholder_data, validation_placeholder_labels = data_shuffler.get_placeholders(name="validation",
                                                                                                    train_dataset=False)
        # Defining a placeholder queue for prefetching
        queue = tf.FIFOQueue(capacity=10,
                             dtypes=[tf.float32, tf.int64],
                             shapes=[train_placeholder_data.get_shape().as_list()[1:], []])

        # Fetching the place holders from the queue
        enqueue_op = queue.enqueue_many([train_placeholder_data, train_placeholder_labels])
        train_feature_batch, train_label_batch = queue.dequeue_many(data_shuffler.train_batch_size)

        # Creating the architecture for train and validation
        if not isinstance(self.architecture, SequenceNetwork):
            raise ValueError("The variable `architecture` must be an instance of "
                             "`bob.learn.tensorflow.network.SequenceNetwork`")

        # Creating graphs
        #train_graph = self.architecture.compute_graph(train_placeholder_data)
        train_graph = self.architecture.compute_graph(train_feature_batch)
        validation_graph = self.architecture.compute_graph(validation_placeholder_data)

        # Defining the loss
        #loss_train = self.loss(train_graph, train_placeholder_labels)
        loss_train = self.loss(train_graph, train_label_batch)
        loss_validation = self.loss(validation_graph, validation_placeholder_labels)

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Learning rate
            batch * data_shuffler.train_batch_size,
            data_shuffler.train_data.shape[0],
            self.weight_decay  # Decay step
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_train,
                                                                              global_step=batch)

        train_prediction = tf.nn.softmax(train_graph)
        validation_prediction = tf.nn.softmax(validation_graph)

        print("Initializing !!")
        # Training
        hdf5 = bob.io.base.HDF5File(os.path.join(self.temp_dir, 'model.hdf5'), 'w')

        with tf.Session() as session:

            tf.initialize_all_variables().run()

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            thread_pool = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=thread_pool)

            threads = start_thread()

            #train_writer = tf.train.SummaryWriter('./LOGS/train',
            #                                      session.graph)

            for step in range(self.iterations):

                try:
                    _, l, lr, _ = session.run([optimizer, loss_train,
                                               learning_rate, train_prediction])

                    if step % self.snapshot == 0:
                        validation_data, validation_labels = data_shuffler.get_batch(train_dataset=False)
                        feed_dict = {validation_placeholder_data: validation_data,
                                     validation_placeholder_labels: validation_labels}##

                        l, predictions = session.run([loss_validation, validation_prediction], feed_dict=feed_dict)
                        accuracy = 100. * numpy.sum(numpy.argmax(predictions, 1) == validation_labels) / predictions.shape[0]

                        print "Step {0}. Loss = {1}, Acc Validation={2}".format(step, l, accuracy)
                except:
                    print "ERROR"
                finally:
                    thread_pool.request_stop()

            #train_writer.close()

            # now they should definetely stop
            thread_pool.request_stop()
            thread_pool.join(threads)
            self.architecture.save(hdf5)
            del hdf5

