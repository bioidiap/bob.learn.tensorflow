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
from bob.learn.tensorflow.datashuffler import OnlineSampling
from bob.learn.tensorflow.utils.session import Session
from .learning_rate import constant

#logger = bob.core.log.setup("bob.learn.tensorflow")

import logging
logger = logging.getLogger("bob.learn")


class Trainer(object):
    """
    One graph trainer.
    Use this trainer when your CNN is composed by one graph

    **Parameters**

    architecture:
      The architecture that you want to run. Should be a :py:class`bob.learn.tensorflow.network.SequenceNetwork`

    optimizer:
      One of the tensorflow optimizers https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html

    use_gpu: bool
      Use GPUs in the training

    loss: :py:class:`bob.learn.tensorflow.loss.BaseLoss`
      Loss function

    temp_dir: str
      The output directory

    learning_rate: `bob.learn.tensorflow.trainers.learning_rate`
      Initial learning rate

    convergence_threshold:

    iterations: int
      Maximum number of iterations

    snapshot: int
      Will take a snapshot of the network at every `n` iterations

    prefetch: bool
      Use extra Threads to deal with the I/O

    model_from_file: str
      If you want to use a pretrained model

    analizer:
      Neural network analizer :py:mod:`bob.learn.tensorflow.analyzers`

    verbosity_level:

    """

    def __init__(self,
                 graph,
                 optimizer=tf.train.AdamOptimizer(),
                 use_gpu=False,
                 loss=None,
                 temp_dir="cnn",

                 # Learning rate
                 learning_rate=None,

                 ###### training options ##########
                 convergence_threshold=0.01,
                 iterations=5000,
                 snapshot=500,
                 validation_snapshot=100,
                 prefetch=False,

                 ## Analizer
                 analizer=SoftmaxAnalizer(),

                 ### Pretrained model
                 model_from_file="",

                 verbosity_level=2):

        #if not isinstance(graph, SequenceNetwork):
        #    raise ValueError("`architecture` should be instance of `SequenceNetwork`")

        self.graph = graph
        self.optimizer_class = optimizer
        self.use_gpu = use_gpu
        self.loss = loss
        self.temp_dir = temp_dir

        if learning_rate is None and model_from_file == "":
            self.learning_rate = constant()
        else:
            self.learning_rate = learning_rate

        self.iterations = iterations
        self.snapshot = snapshot
        self.validation_snapshot = validation_snapshot
        self.convergence_threshold = convergence_threshold
        self.prefetch = prefetch

        # Training variables used in the fit
        self.optimizer = None
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
        self.global_step = None

        self.model_from_file = model_from_file
        self.session = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

    def __del__(self):
        tf.reset_default_graph()

    """
    def compute_graph(self, data_shuffler, prefetch=False, name="", training=True):
        Computes the graph for the trainer.

        ** Parameters **

            data_shuffler: Data shuffler
            prefetch: Uses prefetch
            name: Name of the graph
            training: Is it a training graph?

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
        network_graph = self.architecture.compute_graph(feature_batch, training=training)
        graph = self.loss(network_graph, label_batch)

        return graph
    """

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

    def fit(self, step):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """

        if self.prefetch:
            _, l, lr, summary = self.session.run([self.optimizer, self.training_graph,
                                                  self.learning_rate, self.summaries_train])
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            _, l, lr, summary = self.session.run([self.optimizer, self.training_graph,
                                                  self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    """
    def create_general_summary(self):

        Creates a simple tensorboard summary with the value of the loss and learning rate

        # Train summary
        tf.summary.scalar('loss', self.training_graph)
        tf.summary.scalar('lr', self.learning_rate)
        return tf.summary.merge_all()


    def start_thread(self):

        Start pool of threads for pre-fetching

        **Parameters**
          session: Tensorflow session


        threads = []
        for n in range(3):
            t = threading.Thread(target=self.load_and_enqueue, args=())
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    def load_and_enqueue(self):

        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session


        while not self.thread_pool.should_stop():
            [train_data, train_labels] = self.train_data_shuffler.get_batch()
            [train_placeholder_data, train_placeholder_labels] = self.train_data_shuffler.get_placeholders()

            feed_dict = {train_placeholder_data: train_data,
                         train_placeholder_labels: train_labels}

            self.session.run(self.enqueue_op, feed_dict=feed_dict)

    """

    def bootstrap_graphs_fromfile(self, train_data_shuffler, validation_data_shuffler):
        """
        Bootstrap all the necessary data from file

         ** Parameters **
           session: Tensorflow session
           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation


        """
        saver = self.architecture.load(self.model_from_file, clear_devices=False)

        # Loading training graph
        self.training_graph = tf.get_collection("training_graph")[0]

        # Loding other elements
        self.optimizer = tf.get_collection("optimizer")[0]
        self.learning_rate = tf.get_collection("learning_rate")[0]
        self.summaries_train = tf.get_collection("summaries_train")[0]
        self.global_step = tf.get_collection("global_step")[0]

        if validation_data_shuffler is not None:
            self.validation_graph = tf.get_collection("validation_graph")[0]

        self.bootstrap_placeholders_fromfile(train_data_shuffler, validation_data_shuffler)

        return saver

    def train(self, train_data_shuffler, validation_data_shuffler=None):
        """
        Train the network:

         ** Parameters **

           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation
        """

        # Creating directory
        bob.io.base.create_directories_safe(self.temp_dir)
        self.train_data_shuffler = train_data_shuffler

        logger.info("Initializing !!")

        # Pickle the architecture to save
        #self.architecture.pickle_net(train_data_shuffler.deployment_shape)

        if not isinstance(tf.Tensor, self.graph):
            raise NotImplemented("Not tensor still not implemented")

        self.session = Session.instance(new=True).session

        # Loading a pretrained model
        if self.model_from_file != "":
            logger.info("Loading pretrained model from {0}".format(self.model_from_file))
            saver = self.bootstrap_graphs_fromfile(train_data_shuffler, validation_data_shuffler)

            start_step = self.global_step.eval(session=self.session)

        else:
            start_step = 0

            # TODO: find an elegant way to provide this as a parameter of the trainer
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            tf.add_to_collection("global_step", self.global_step)

            # Preparing the optimizer
            self.optimizer_class._learning_rate = self.learning_rate
            self.optimizer = self.optimizer_class.minimize(self.training_graph, global_step=self.global_step)
            tf.add_to_collection("optimizer", self.optimizer)
            tf.add_to_collection("learning_rate", self.learning_rate)

            # Train summary
            tf.global_variables_initializer().run(session=self.session)

            # Original tensorflow saver object
            saver = tf.train.Saver(var_list=tf.global_variables())

        #if isinstance(train_data_shuffler, OnlineSampling):
        #    train_data_shuffler.set_feature_extractor(self.architecture, session=self.session)

        # Start a thread to enqueue data asynchronously, and hide I/O latency.
        #if self.prefetch:
        #    self.thread_pool = tf.train.Coordinator()
        #    tf.train.start_queue_runners(coord=self.thread_pool, sess=self.session)
        #    threads = self.start_thread()

        # TENSOR BOARD SUMMARY
        self.train_summary_writter = tf.summary.FileWriter(os.path.join(self.temp_dir, 'train'), self.session.graph)
        for step in range(start_step, self.iterations):
            start = time.time()
            self.fit(step)
            end = time.time()
            summary = summary_pb2.Summary.Value(tag="elapsed_time", simple_value=float(end-start))
            self.train_summary_writter.add_summary(summary_pb2.Summary(value=[summary]), step)

            # Running validation
            #if validation_data_shuffler is not None and step % self.validation_snapshot == 0:
            #    self.compute_validation(validation_data_shuffler, step)

            #    if self.analizer is not None:
            #        self.validation_summary_writter.add_summary(self.analizer(
            #             validation_data_shuffler, self.architecture, self.session), step)

            # Taking snapshot
            if step % self.snapshot == 0:
                logger.info("Taking snapshot")
                path = os.path.join(self.temp_dir, 'model_snapshot{0}.ckp'.format(step))
                self.architecture.save(saver, path)

        logger.info("Training finally finished")

        self.train_summary_writter.close()
        if validation_data_shuffler is not None:
            self.validation_summary_writter.close()

        # Saving the final network
        path = os.path.join(self.temp_dir, 'model.ckp')
        self.architecture.save(saver, path)

        if self.prefetch:
            # now they should definetely stop
            self.thread_pool.request_stop()
            self.thread_pool.join(threads)
