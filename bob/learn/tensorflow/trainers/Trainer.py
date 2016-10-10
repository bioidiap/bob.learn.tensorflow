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
import bob.core
from tensorflow.core.framework import summary_pb2

logger = bob.core.log.setup("bob.learn.tensorflow")

class Trainer(object):

    def __init__(self,
                 architecture,
                 optimizer=tf.train.AdamOptimizer(),
                 use_gpu=False,
                 loss=None,
                 temp_dir="cnn",

                 # Learning rate
                 base_learning_rate=0.001,
                 weight_decay=0.9,

                 ###### training options ##########
                 convergence_threshold=0.01,
                 iterations=5000,
                 snapshot=100,
                 prefetch=False,
                 verbosity_level=2):
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
        self.optimizer_class = optimizer
        self.use_gpu = use_gpu
        self.loss = loss
        self.temp_dir = temp_dir

        self.base_learning_rate = base_learning_rate
        self.weight_decay = weight_decay

        self.iterations = iterations
        self.snapshot = snapshot
        self.convergence_threshold = convergence_threshold
        self.prefetch = prefetch

        # Training variables used in the fit
        self.optimizer = None
        self.training_graph = None
        self.learning_rate = None
        self.training_graph = None
        self.train_data_shuffler = None
        self.summaries_train = None
        self.train_summary_writter = None

        # Validation data
        self.validation_graph = None
        self.validation_summary_writter = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

    def compute_graph(self, data_shuffler, name=""):
        """
        Computes the graph for the trainer.

        ** Parameters **

            data_shuffler: Data shuffler
            name: Name of the graph
        """

        # Defining place holders
        if self.prefetch:
            placeholder_data, placeholder_labels = data_shuffler.get_placeholders_forprefetch(name=name)

            #if validation_data_shuffler is not None:
            #    validation_placeholder_data, validation_placeholder_labels = \
            #        validation_data_shuffler.get_placeholders(name="validation")

            # Defining a placeholder queue for prefetching
            queue = tf.FIFOQueue(capacity=10,
                                 dtypes=[tf.float32, tf.int64],
                                 shapes=[placeholder_data.get_shape().as_list()[1:], []])

            # Fetching the place holders from the queue
            enqueue_op = queue.enqueue_many([placeholder_data, placeholder_labels])
            feature_batch, label_batch = queue.dequeue_many(data_shuffler.batch_size)

            # Creating the architecture for train and validation
            if not isinstance(self.architecture, SequenceNetwork):
                raise ValueError("The variable `architecture` must be an instance of "
                                 "`bob.learn.tensorflow.network.SequenceNetwork`")
        else:
            feature_batch, label_batch = data_shuffler.get_placeholders(name=name)

        # Creating graphs and defining the loss
        network_graph = self.architecture.compute_graph(feature_batch)
        graph = self.loss(network_graph, label_batch)

        return graph

    def get_feed_dict(self, data_shuffler):
        """
        Computes the feed_dict for the graph

        ** Parameters **

            data_shuffler:

        """
        data, labels = data_shuffler.get_batch()
        data_placeholder, label_placeholder = data_shuffler.get_placeholders()

        feed_dict = {data_placeholder: data,
                     label_placeholder: labels}
        return feed_dict

    def __fit(self, session, step):
        if self.prefetch:
            raise ValueError("Prefetch not implemented for such trainer")
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            _, l, lr, summary = session.run([self.optimizer, self.training_graph,
                                             self.learning_rate, self.summaries_train], feed_dict=feed_dict)

            logger.info("Loss training set step={0} = {1}".format(step, l))
            self.train_summary_writter.add_summary(summary, step)

    def __compute_validation(self, session, data_shuffler, step):

        if self.validation_summary_writter is None:
            self.validation_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'validation'), session.graph)

        self.validation_graph = self.compute_graph(data_shuffler, name="validation")
        feed_dict = self.get_feed_dict(data_shuffler)
        l = session.run(self.validation_graph, feed_dict=feed_dict)

        summaries = []
        summaries.append(summary_pb2.Summary.Value(tag="loss", simple_value=float(l)))
        self.validation_summary_writter.add_summary(summary_pb2.Summary(value=summaries), step)
        logger.info("Loss VALIDATION set step={0} = {1}".format(step, l))

    def __create_general_summary(self):
        # Train summary
        tf.scalar_summary('loss', self.training_graph, name="train")
        tf.scalar_summary('lr', self.learning_rate, name="train")
        return tf.merge_all_summaries()


    """
    def start_thread(self):
        threads = []
        for n in range(1):
            t = threading.Thread(target=self.load_and_enqueue)
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


    def load_and_enqueue(self):
        Injecting data in the place holder queue


        #while not thread_pool.should_stop():
        #for i in range(self.iterations):
        while not thread_pool.should_stop():
            train_data, train_labels = train_data_shuffler.get_batch()

            feed_dict = {train_placeholder_data: train_data,
                         train_placeholder_labels: train_labels}

            session.run(enqueue_op, feed_dict=feed_dict)

    """

    def train(self, train_data_shuffler, validation_data_shuffler=None):
        """
        Do the loop forward --> backward --|
                      ^--------------------|
        """

        # Creating directory
        bob.io.base.create_directories_safe(self.temp_dir)
        self.train_data_shuffler = train_data_shuffler

        # TODO: find an elegant way to provide this as a parameter of the trainer
        self.learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,  # Learning rate
            train_data_shuffler.batch_size,
            train_data_shuffler.n_samples,
            self.weight_decay  # Decay step
        )

        self.training_graph = self.compute_graph(train_data_shuffler, name="train")

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.training_graph, global_step=tf.Variable(0))

        # Train summary
        self.summaries_train = self.__create_general_summary()

        logger.info("Initializing !!")
        # Training
        hdf5 = bob.io.base.HDF5File(os.path.join(self.temp_dir, 'model.hdf5'), 'w')

        with tf.Session() as session:

            tf.initialize_all_variables().run()

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            #thread_pool = tf.train.Coordinator()
            #tf.train.start_queue_runners(coord=thread_pool)
            #threads = start_thread()

            # TENSOR BOARD SUMMARY
            self.train_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'train'), session.graph)

            for step in range(self.iterations):

                self.__fit(session, step)
                if validation_data_shuffler is not None and step % self.snapshot == 0:
                    self.__compute_validation(session, validation_data_shuffler, step)


                #    validation_data, validation_labels = validation_data_shuffler.get_batch()

                #    feed_dict = {validation_placeholder_data: validation_data,
                #                 validation_placeholder_labels: validation_labels}

                    #l, predictions = session.run([loss_validation, validation_prediction, ], feed_dict=feed_dict)
                    #l, summary = session.run([loss_validation, merged_validation], feed_dict=feed_dict)
                    #import ipdb; ipdb.set_trace();
                #    l = session.run(loss_validation, feed_dict=feed_dict)
                #    summaries = []
                #    summaries.append(summary_pb2.Summary.Value(tag="loss", simple_value=float(l)))
                #    validation_writer.add_summary(summary_pb2.Summary(value=summaries), step)

                    #l = session.run([loss_validation], feed_dict=feed_dict)
                    #accuracy = 100. * numpy.sum(numpy.argmax(predictions, 1) == validation_labels) / predictions.shape[0]
                    #validation_writer.add_summary(summary, step)
                    #print "Step {0}. Loss = {1}, Acc Validation={2}".format(step, l, accuracy)
                #    print "Step {0}. Loss = {1}".format(step, l)

            logger.info("Training finally finished")

            self.train_summary_writter.close()
            if validation_data_shuffler is not None:
                self.validation_summary_writter.close()

            self.architecture.save(hdf5)
            del hdf5



            # now they should definetely stop
            #thread_pool.request_stop()
            #thread_pool.join(threads)

