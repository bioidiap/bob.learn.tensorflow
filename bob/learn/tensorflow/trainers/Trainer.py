#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
from ..network import SequenceNetwork
import threading
import os
import bob.io.base
import bob.core
from ..analyzers import SoftmaxAnalizer
from tensorflow.core.framework import summary_pb2
import time
from bob.learn.tensorflow.datashuffler.OnlineSampling import OnLineSampling


logger = bob.core.log.setup("bob.learn.tensorflow")


class Trainer(object):
    """
    One graph trainer.
    Use this trainer when your CNN is composed by one graph

    **Parameters**
      architecture: The architecture that you want to run. Should be a :py:class`bob.learn.tensorflow.network.SequenceNetwork`
      optimizer: One of the tensorflow optimizers https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
      use_gpu: Use GPUs in the training
      loss: Loss
      temp_dir: The output directory

      base_learning_rate: Initial learning rate
      weight_decay:
      convergence_threshold:

      iterations: Maximum number of iterations
      snapshot: Will take a snapshot of the network at every `n` iterations
      prefetch: Use extra Threads to deal with the I/O
      analizer: Neural network analizer :py:mod:`bob.learn.tensorflow.analyzers`
      verbosity_level:

    """
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

                 ## Analizer
                 analizer=SoftmaxAnalizer(),

                 verbosity_level=2):

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
        self.thread_pool = None

        # Validation data
        self.validation_graph = None
        self.validation_summary_writter = None

        # Analizer
        self.analizer = analizer

        self.thread_pool = None
        self.enqueue_op = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

    def __del__(self):
        tf.reset_default_graph()

    def compute_graph(self, data_shuffler, prefetch=False, name=""):
        """
        Computes the graph for the trainer.


        ** Parameters **

            data_shuffler: Data shuffler
            prefetch:
            name: Name of the graph
        """

        # Defining place holders
        if prefetch:
            [placeholder_data, placeholder_labels] = data_shuffler.get_placeholders_forprefetch(name=name)

            # Defining a placeholder queue for prefetching
            queue = tf.FIFOQueue(capacity=10,
                                 dtypes=[tf.float32, tf.int64],
                                 shapes=[placeholder_data.get_shape().as_list()[1:], []])

            # Fetching the place holders from the queue
            self.enqueue_op = queue.enqueue_many([placeholder_data, placeholder_labels])
            feature_batch, label_batch = queue.dequeue_many(data_shuffler.batch_size)

            # Creating the architecture for train and validation
            if not isinstance(self.architecture, SequenceNetwork):
                raise ValueError("The variable `architecture` must be an instance of "
                                 "`bob.learn.tensorflow.network.SequenceNetwork`")
        else:
            [feature_batch, label_batch] = data_shuffler.get_placeholders(name=name)

        # Creating graphs and defining the loss
        network_graph = self.architecture.compute_graph(feature_batch)
        graph = self.loss(network_graph, label_batch)

        return graph

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """
        [data, labels] = data_shuffler.get_batch()
        [data_placeholder, label_placeholder] = data_shuffler.get_placeholders()

        feed_dict = {data_placeholder: data,
                     label_placeholder: labels}
        return feed_dict

    def fit(self, session, step):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """

        if self.prefetch:
            _, l, lr, summary = session.run([self.optimizer, self.training_graph,
                                             self.learning_rate, self.summaries_train])
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            _, l, lr, summary = session.run([self.optimizer, self.training_graph,
                                             self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    def compute_validation(self,  session, data_shuffler, step):
        """
        Computes the loss in the validation set

        ** Parameters **
            session: Tensorflow session
            data_shuffler: The data shuffler to be used
            step: Iteration number

        """
        # Opening a new session for validation
        self.validation_graph = self.compute_graph(data_shuffler, name="validation")
        feed_dict = self.get_feed_dict(data_shuffler)
        l = session.run(self.validation_graph, feed_dict=feed_dict)

        if self.validation_summary_writter is None:
            self.validation_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'validation'), session.graph)

        summaries = []
        summaries.append(summary_pb2.Summary.Value(tag="loss", simple_value=float(l)))
        self.validation_summary_writter.add_summary(summary_pb2.Summary(value=summaries), step)
        logger.info("Loss VALIDATION set step={0} = {1}".format(step, l))

    def create_general_summary(self):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """

        # Train summary
        tf.scalar_summary('loss', self.training_graph, name="train")
        tf.scalar_summary('lr', self.learning_rate, name="train")
        return tf.merge_all_summaries()

    def start_thread(self, session):
        """
        Start pool of threads for pre-fetching

        **Parameters**
          session: Tensorflow session
        """

        threads = []
        for n in range(3):
            t = threading.Thread(target=self.load_and_enqueue, args=(session,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    def load_and_enqueue(self, session):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session
        """

        while not self.thread_pool.should_stop():
            [train_data, train_labels] = self.train_data_shuffler.get_batch()
            [train_placeholder_data, train_placeholder_labels] = self.train_data_shuffler.get_placeholders()

            feed_dict = {train_placeholder_data: train_data,
                         train_placeholder_labels: train_labels}

            session.run(self.enqueue_op, feed_dict=feed_dict)

    def train(self, train_data_shuffler, validation_data_shuffler=None):
        """
        Train the network
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

        self.training_graph = self.compute_graph(train_data_shuffler, prefetch=self.prefetch, name="train")

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.training_graph, global_step=tf.Variable(0))

        # Train summary
        self.summaries_train = self.create_general_summary()

        logger.info("Initializing !!")
        # Training
        hdf5 = bob.io.base.HDF5File(os.path.join(self.temp_dir, 'model.hdf5'), 'w')

        with tf.Session() as session:

            tf.initialize_all_variables().run()

            if isinstance(train_data_shuffler, OnLineSampling):
                train_data_shuffler.set_feature_extractor(self.architecture, session=session)

            # Start a thread to enqueue data asynchronously, and hide I/O latency.
            if self.prefetch:
                self.thread_pool = tf.train.Coordinator()
                tf.train.start_queue_runners(coord=self.thread_pool)
                threads = self.start_thread(session)

            # TENSOR BOARD SUMMARY
            self.train_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'train'), session.graph)
            for step in range(self.iterations):

                start = time.time()
                self.fit(session, step)
                end = time.time()
                summary = summary_pb2.Summary.Value(tag="elapsed_time", simple_value=float(end-start))
                self.train_summary_writter.add_summary(summary_pb2.Summary(value=[summary]), step)

                if validation_data_shuffler is not None and step % self.snapshot == 0:
                    self.compute_validation(session, validation_data_shuffler, step)

                    if self.analizer is not None:
                        self.validation_summary_writter.add_summary(self.analizer(
                             validation_data_shuffler, self.architecture, session), step)

            logger.info("Training finally finished")

            self.train_summary_writter.close()
            if validation_data_shuffler is not None:
                self.validation_summary_writter.close()

            self.architecture.save(hdf5)
            del hdf5

            if self.prefetch:
                # now they should definetely stop
                self.thread_pool.request_stop()
                self.thread_pool.join(threads)

            session.close() # For some reason the session is not closed after the context manager finishes
