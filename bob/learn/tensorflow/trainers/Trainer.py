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
                 train_data_shuffler,

                 ###### training options ##########
                 iterations=5000,
                 snapshot=500,
                 validation_snapshot=100,

                 ## Analizer
                 analizer=SoftmaxAnalizer(),

                 # Temporatu dir
                 temp_dir="cnn",

                 verbosity_level=2):

        self.train_data_shuffler = train_data_shuffler
        self.temp_dir = temp_dir

        self.iterations = iterations
        self.snapshot = snapshot
        self.validation_snapshot = validation_snapshot

        # Training variables used in the fit
        self.summaries_train = None
        self.train_summary_writter = None
        self.thread_pool = None

        # Validation data
        self.validation_summary_writter = None

        # Analizer
        self.analizer = analizer
        self.global_step = None

        self.session = None

        self.graph = None
        self.loss = None
        self.predictor = None
        self.optimizer_class = None
        self.learning_rate = None
        # Training variables used in the fit
        self.optimizer = None
        self.data_ph = None
        self.label_ph = None
        self.saver = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

        # Creating the session
        self.session = Session.instance(new=True).session
        self.from_scratch = True

    def create_network_from_scratch(self,
                                    graph,
                                    optimizer=tf.train.AdamOptimizer(),
                                    loss=None,

                                    # Learning rate
                                    learning_rate=None,
                                    ):

        self.data_ph = self.train_data_shuffler("data")
        self.label_ph = self.train_data_shuffler("label")
        self.graph = graph
        self.loss = loss
        self.predictor = self.loss(self.graph, self.train_data_shuffler("label", from_queue=True))

        self.optimizer_class = optimizer
        self.learning_rate = learning_rate

        # TODO: find an elegant way to provide this as a parameter of the trainer
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Saving all the variables
        self.saver = tf.train.Saver(var_list=tf.global_variables())

        tf.add_to_collection("global_step", self.global_step)

        tf.add_to_collection("graph", self.graph)
        tf.add_to_collection("predictor", self.predictor)

        tf.add_to_collection("data_ph", self.data_ph)
        tf.add_to_collection("label_ph", self.label_ph)

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.predictor, global_step=self.global_step)
        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        self.summaries_train = self.create_general_summary()
        tf.add_to_collection("summaries_train", self.summaries_train)


        # Creating the variables
        tf.global_variables_initializer().run(session=self.session)

    def create_network_from_file(self, model_from_file):
        """
        Bootstrap all the necessary data from file

         ** Parameters **
           session: Tensorflow session
           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation

        """
        #saver = self.architecture.load(self.model_from_file, clear_devices=False)
        self.saver = tf.train.import_meta_graph(model_from_file + ".meta")
        self.saver.restore(self.session, model_from_file)

        # Loading training graph
        self.data_ph = tf.get_collection("data_ph")[0]
        self.label_ph = tf.get_collection("label_ph")[0]

        self.graph = tf.get_collection("graph")[0]
        self.predictor = tf.get_collection("predictor")[0]

        # Loding other elements
        self.optimizer = tf.get_collection("optimizer")[0]
        self.learning_rate = tf.get_collection("learning_rate")[0]
        self.summaries_train = tf.get_collection("summaries_train")[0]
        self.global_step = tf.get_collection("global_step")[0]
        self.from_scratch = False

        # Creating the variables
        #tf.global_variables_initializer().run(session=self.session)

    def __del__(self):
        tf.reset_default_graph()

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """
        [data, labels] = data_shuffler.get_batch()

        feed_dict = {self.data_ph: data,
                     self.label_ph: labels}
        return feed_dict

    def fit(self, step):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """

        if self.train_data_shuffler.prefetch:
            _, l, lr, summary = self.session.run([self.optimizer, self.predictor,
                                                  self.learning_rate, self.summaries_train])
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            _, l, lr, summary = self.session.run([self.optimizer, self.predictor,
                                                  self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    def compute_validation(self, data_shuffler, step):
        """
        Computes the loss in the validation set

        ** Parameters **
            session: Tensorflow session
            data_shuffler: The data shuffler to be used
            step: Iteration number

        """
        pass
        # Opening a new session for validation
        #feed_dict = self.get_feed_dict(data_shuffler)
        #l, summary = self.session.run(self.predictor, self.summaries_train, feed_dict=feed_dict)
        #train_summary_writter.add_summary(summary, step)


        #summaries = [summary_pb2.Summary.Value(tag="loss", simple_value=float(l))]
        #self.validation_summary_writter.add_summary(summary_pb2.Summary(value=summaries), step)
        #logger.info("Loss VALIDATION set step={0} = {1}".format(step, l))

    def create_general_summary(self):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """

        # Train summary
        tf.summary.scalar('loss', self.predictor)
        tf.summary.scalar('lr', self.learning_rate)
        return tf.summary.merge_all()

    def start_thread(self):
        """
        Start pool of threads for pre-fetching

        **Parameters**
          session: Tensorflow session
        """

        threads = []
        for n in range(3):
            t = threading.Thread(target=self.load_and_enqueue, args=())
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    def load_and_enqueue(self):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session

        """
        while not self.thread_pool.should_stop():
            [train_data, train_labels] = self.train_data_shuffler.get_batch()

            feed_dict = {self.data_ph: train_data,
                         self.label_ph: train_labels}

            self.session.run(self.inputs.enqueue_op, feed_dict=feed_dict)

    def train(self, validation_data_shuffler=None):
        """
        Train the network:

         ** Parameters **

           train_data_shuffler: Data shuffler for training
           validation_data_shuffler: Data shuffler for validation
        """

        # Creating directory
        bob.io.base.create_directories_safe(self.temp_dir)

        logger.info("Initializing !!")

        # Loading a pretrained model
        if self.from_scratch:
            start_step = 0
        else:
            start_step = self.global_step.eval(session=self.session)

        #if isinstance(train_data_shuffler, OnlineSampling):
        #    train_data_shuffler.set_feature_extractor(self.architecture, session=self.session)

        # Start a thread to enqueue data asynchronously, and hide I/O latency.
        if self.train_data_shuffler.prefetch:
            self.thread_pool = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=self.thread_pool, sess=self.session)
            threads = self.start_thread()

        # TENSOR BOARD SUMMARY
        self.train_summary_writter = tf.summary.FileWriter(os.path.join(self.temp_dir, 'train'), self.session.graph)
        if validation_data_shuffler is not None:
            self.validation_summary_writter = tf.summary.FileWriter(os.path.join(self.temp_dir, 'validation'),
                                                                    self.session.graph)
        # Loop for
        for step in range(start_step, self.iterations):
            # Run fit in the graph
            start = time.time()
            self.fit(step)
            end = time.time()
            summary = summary_pb2.Summary.Value(tag="elapsed_time", simple_value=float(end-start))
            self.train_summary_writter.add_summary(summary_pb2.Summary(value=[summary]), step)

            # Running validation
            if validation_data_shuffler is not None and step % self.validation_snapshot == 0:
                self.compute_validation(validation_data_shuffler, step)

                #if self.analizer is not None:
                #    self.validation_summary_writter.add_summary(self.analizer(
                #         validation_data_shuffler, self.architecture, self.session), step)

            # Taking snapshot
            if step % self.snapshot == 0:
                logger.info("Taking snapshot")
                path = os.path.join(self.temp_dir, 'model_snapshot{0}.ckp'.format(step))
                self.saver.save(self.session, path, global_step=step)
                #self.architecture.save(saver, path)

        logger.info("Training finally finished")

        self.train_summary_writter.close()
        if validation_data_shuffler is not None:
            self.validation_summary_writter.close()

        # Saving the final network
        path = os.path.join(self.temp_dir, 'model.ckp')
        self.saver.save(self.session, path)

        if self.train_data_shuffler.prefetch:
            # now they should definetely stop
            self.thread_pool.request_stop()
            #self.thread_pool.join(threads)
