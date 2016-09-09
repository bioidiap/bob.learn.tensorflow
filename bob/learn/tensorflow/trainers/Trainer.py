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
                 architecture,
                 optimizer=tf.train.AdamOptimizer(),
                 use_gpu=False,
                 loss=None,
                 temp_dir="",

                 # Learning rate
                 base_learning_rate=0.001,
                 weight_decay=0.9,

                 ###### training options ##########
                 convergence_threshold=0.01,
                 iterations=5000,
                 snapshot=100):
        """

        **Parameters**
          architecture: The architecture that you want to run. Should be a :py:class`bob.learn.tensorflow.network.SequenceNetwork`
          optimizer: One of the tensorflow optimizers https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
          use_gpu: Use GPUs in the training
          loss: Loss
          temp_dir:
          iterations:
          snapshot:
          convergence_threshold:
        """
        if not isinstance(architecture, SequenceNetwork):
            raise ValueError("`architecture` should be instance of `SequenceNetwork`")

        self.architecture = architecture
        self.optimizer = optimizer
        self.use_gpu = use_gpu
        self.loss = loss
        self.temp_dir = temp_dir

        self.base_learning_rate = base_learning_rate
        self.weight_decay = weight_decay

        self.iterations = iterations
        self.snapshot = snapshot
        self.convergence_threshold = convergence_threshold

    def train(self, train_data_shuffler, validation_data_shuffler=None):
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
                train_data, train_labels = train_data_shuffler.get_batch()

                feed_dict = {train_placeholder_data: train_data,
                             train_placeholder_labels: train_labels}

                session.run(enqueue_op, feed_dict=feed_dict)

        # TODO: find an elegant way to provide this as a parameter of the trainer
        learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,  # Learning rate
            train_data_shuffler.batch_size,
            train_data_shuffler.n_samples,
            self.weight_decay  # Decay step
        )

        # Defining place holders
        train_placeholder_data, train_placeholder_labels = train_data_shuffler.get_placeholders_forprefetch(name="train")
        if validation_data_shuffler is not None:
            validation_placeholder_data, validation_placeholder_labels = \
                validation_data_shuffler.get_placeholders(name="validation")
        # Defining a placeholder queue for prefetching
        queue = tf.FIFOQueue(capacity=10,
                             dtypes=[tf.float32, tf.int64],
                             shapes=[train_placeholder_data.get_shape().as_list()[1:], []])

        # Fetching the place holders from the queue
        enqueue_op = queue.enqueue_many([train_placeholder_data, train_placeholder_labels])
        train_feature_batch, train_label_batch = queue.dequeue_many(train_data_shuffler.batch_size)

        # Creating the architecture for train and validation
        if not isinstance(self.architecture, SequenceNetwork):
            raise ValueError("The variable `architecture` must be an instance of "
                             "`bob.learn.tensorflow.network.SequenceNetwork`")

        # Creating graphs and defining the loss
        train_graph = self.architecture.compute_graph(train_feature_batch)
        loss_train = self.loss(train_graph, train_label_batch)
        train_prediction = tf.nn.softmax(train_graph)
        if validation_data_shuffler is not None:
            validation_graph = self.architecture.compute_graph(validation_placeholder_data)
            loss_validation = self.loss(validation_graph, validation_placeholder_labels)
            validation_prediction = tf.nn.softmax(validation_graph)

        # Preparing the optimizer
        self.optimizer._learning_rate = learning_rate
        optimizer = self.optimizer.minimize(loss_train, global_step=tf.Variable(0))

        print("Initializing !!")
        # Training
        hdf5 = bob.io.base.HDF5File(os.path.join(self.temp_dir, 'model.hdf5'), 'w')

        with tf.Session() as session:

            tf.initialize_all_variables().run()

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            thread_pool = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=thread_pool)

            threads = start_thread()

            train_writer = tf.train.SummaryWriter('./LOGS/train', session.graph)

            for step in range(self.iterations):

                _, l, lr, _ = session.run([optimizer, loss_train,
                                           learning_rate, train_prediction])

                if validation_data_shuffler is not None and step % self.snapshot == 0:
                    validation_data, validation_labels = validation_data_shuffler.get_batch()

                    feed_dict = {validation_placeholder_data: validation_data,
                                 validation_placeholder_labels: validation_labels}

                    l, predictions = session.run([loss_validation, validation_prediction], feed_dict=feed_dict)
                    accuracy = 100. * numpy.sum(numpy.argmax(predictions, 1) == validation_labels) / predictions.shape[0]

                    print "Step {0}. Loss = {1}, Acc Validation={2}".format(step, l, accuracy)

            train_writer.close()

            # now they should definetely stop
            thread_pool.request_stop()
            thread_pool.join(threads)
            self.architecture.save(hdf5)
            del hdf5

