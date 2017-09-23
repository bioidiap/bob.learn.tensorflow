#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
import threading
import os
import bob.io.base
import bob.core
from ..analyzers import SoftmaxAnalizer
from tensorflow.core.framework import summary_pb2
import time
from bob.learn.tensorflow.datashuffler import OnlineSampling, TFRecord
from bob.learn.tensorflow.utils.session import Session
from .learning_rate import constant
import time

#logger = bob.core.log.setup("bob.learn.tensorflow")

import logging
logger = logging.getLogger("bob.learn")


class Trainer(object):
    """
    One graph trainer.
    Use this trainer when your CNN is composed by one graph

    **Parameters**

    train_data_shuffler:
      The data shuffler used for batching data for training

    iterations:
      Maximum number of iterations

    snapshot:
      Will take a snapshot of the network at every `n` iterations

    validation_snapshot:
      Test with validation each `n` iterations

    analizer:
      Neural network analizer :py:mod:`bob.learn.tensorflow.analyzers`

    temp_dir: str
      The output directory

    verbosity_level:

    """

    def __init__(self,
                 train_data_shuffler,
                 validation_data_shuffler=None,

                 ###### training options ##########
                 iterations=5000,
                 snapshot=500,
                 validation_snapshot=100,
                 keep_checkpoint_every_n_hours=2,

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
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        # Training variables used in the fit
        self.summaries_train = None
        self.train_summary_writter = None
        self.thread_pool = None

        # Validation data
        self.validation_summary_writter = None
        self.summaries_validation = None
        self.validation_data_shuffler = validation_data_shuffler

        # Analizer
        self.analizer = analizer
        self.global_step = None

        self.session = None

        self.graph = None
        self.validation_graph = None
                
        self.loss = None
        
        self.predictor = None
        self.validation_predictor = None        
        
        self.optimizer_class = None
        self.learning_rate = None

        # Training variables used in the fit
        self.optimizer = None
        
        self.data_ph = None
        self.label_ph = None
        
        self.validation_data_ph = None
        self.validation_label_ph = None
        
        self.saver = None

        bob.core.log.set_verbosity_level(logger, verbosity_level)

        # Creating the session
        self.session = Session.instance(new=True).session
        self.from_scratch = True
        
    def train(self):
        """
        Train the network        
        Here we basically have the loop for that takes your graph and do a sequence of session.run
        """

        # Creating directories
        bob.io.base.create_directories_safe(self.temp_dir)
        logger.info("Initializing !!")

        # Loading a pretrained model
        if self.from_scratch:
            start_step = 0
        else:
            start_step = self.global_step.eval(session=self.session)

        # TODO: Put this back as soon as possible
        #if isinstance(train_data_shuffler, OnlineSampling):
        #    train_data_shuffler.set_feature_extractor(self.architecture, session=self.session)

        # Start a thread to enqueue data asynchronously, and hide I/O latency.        
        if self.train_data_shuffler.prefetch:
            self.thread_pool = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=self.thread_pool, sess=self.session)
            # In case you have your own queue
            if not isinstance(self.train_data_shuffler, TFRecord):
                threads = self.start_thread()

        # Bootstrapping the summary writters
        self.train_summary_writter = tf.summary.FileWriter(os.path.join(self.temp_dir, 'train'), self.session.graph)
        if self.validation_data_shuffler is not None:
            self.validation_summary_writter = tf.summary.FileWriter(os.path.join(self.temp_dir, 'validation'),
                                                                    self.session.graph)

        ######################### Loop for #################
        for step in range(start_step, start_step+self.iterations):
            # Run fit in the graph
            start = time.time()
            self.fit(step)
            end = time.time()

            summary = summary_pb2.Summary.Value(tag="elapsed_time", simple_value=float(end-start))
            self.train_summary_writter.add_summary(summary_pb2.Summary(value=[summary]), step)

            # Running validation
            if self.validation_data_shuffler is not None and step % self.validation_snapshot == 0:
                self.compute_validation(step)

            # Taking snapshot
            if step % self.snapshot == 0:
                logger.info("Taking snapshot")
                path = os.path.join(self.temp_dir, 'model_snapshot{0}.ckp'.format(step))
                self.saver.save(self.session, path, global_step=step)

        # Running validation for the last time
        if self.validation_data_shuffler is not None:
            self.compute_validation(step)
            
        logger.info("Training finally finished")

        self.train_summary_writter.close()
        if self.validation_data_shuffler is not None:
            self.validation_summary_writter.close()

        # Saving the final network
        path = os.path.join(self.temp_dir, 'model.ckp')
        self.saver.save(self.session, path)

        if self.train_data_shuffler.prefetch or isinstance(self.train_data_shuffler, TFRecord):
            # now they should definetely stop
            self.thread_pool.request_stop()
            #if not isinstance(self.train_data_shuffler, TFRecord):
            #    self.thread_pool.join(threads)        

    def create_network_from_scratch(self,
                                    graph,
                                    validation_graph=None,
                                    optimizer=tf.train.AdamOptimizer(),
                                    loss=None,

                                    # Learning rate
                                    learning_rate=None,
                                    ):

        """
        Prepare all the tensorflow variables before training.
        
        **Parameters**

            graph: Input graph for training

            optimizer: Solver

            loss: Loss function

            learning_rate: Learning rate
        """

        # Getting the pointer to the placeholders
        self.data_ph = self.train_data_shuffler("data", from_queue=True)
        self.label_ph = self.train_data_shuffler("label", from_queue=True)
                
        self.graph = graph
        self.loss = loss        

        # Attaching the loss in the graph
        self.predictor = self.loss(self.graph, self.label_ph)
        
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.predictor, global_step=self.global_step)

        # Saving all the variables
        self.saver = tf.train.Saver(var_list=tf.global_variables() + tf.local_variables(), 
                                    keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

        self.summaries_train = self.create_general_summary(self.predictor, self.graph, self.label_ph)

        # SAving some variables
        tf.add_to_collection("global_step", self.global_step)
        tf.add_to_collection("graph", self.graph)
        tf.add_to_collection("predictor", self.predictor)

        tf.add_to_collection("data_ph", self.data_ph)
        tf.add_to_collection("label_ph", self.label_ph)

        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        tf.add_to_collection("summaries_train", self.summaries_train)

        # Same business with the validation
        if self.validation_data_shuffler is not None:
            self.validation_data_ph = self.validation_data_shuffler("data", from_queue=True)
            self.validation_label_ph = self.validation_data_shuffler("label", from_queue=True)

            self.validation_graph = validation_graph

            self.validation_predictor = self.loss(self.validation_graph, self.validation_label_ph)

            self.summaries_validation = self.create_general_summary(self.validation_predictor, self.validation_graph, self.validation_label_ph)
            tf.add_to_collection("summaries_validation", self.summaries_validation)
            
            tf.add_to_collection("validation_graph", self.validation_graph)
            tf.add_to_collection("validation_data_ph", self.validation_data_ph)
            tf.add_to_collection("validation_label_ph", self.validation_label_ph)

            tf.add_to_collection("validation_predictor", self.validation_predictor)
            tf.add_to_collection("summaries_validation", self.summaries_validation)

        # Creating the variables
        tf.local_variables_initializer().run(session=self.session)
        tf.global_variables_initializer().run(session=self.session)

    def load_checkpoint(self, file_name, clear_devices=True):
        """
        Load a checkpoint

        ** Parameters **

           file_name:
                Name of the metafile to be loaded.
                If a directory is passed, the last checkpoint will be loaded

        """
        if os.path.isdir(file_name):
            checkpoint_path = tf.train.get_checkpoint_state(file_name).model_checkpoint_path
            self.saver = tf.train.import_meta_graph(checkpoint_path + ".meta", clear_devices=clear_devices)
            self.saver.restore(self.session, tf.train.latest_checkpoint(file_name))
        else:
            self.saver = tf.train.import_meta_graph(file_name, clear_devices=clear_devices)
            self.saver.restore(self.session, tf.train.latest_checkpoint(os.path.dirname(file_name)))

    def create_network_from_file(self, file_name, clear_devices=True):
        """
        Bootstrap a graph from a checkpoint

         ** Parameters **

           file_name: Name of of the checkpoing
        """

        logger.info("Loading last checkpoint !!")
        self.load_checkpoint(file_name, clear_devices=True)

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
        
        # Loading the validation bits
        if self.validation_data_shuffler is not None:
            self.summaries_validation = tf.get_collection("summaries_validation")[0]

            self.validation_graph = tf.get_collection("validation_graph")[0]
            self.validation_data_ph = tf.get_collection("validation_data_ph")[0]
            self.validation_label = tf.get_collection("validation_label_ph")[0]

            self.validation_predictor = tf.get_collection("validation_predictor")[0]
            self.summaries_validation = tf.get_collection("summaries_validation")[0]

    def __del__(self):
        tf.reset_default_graph()

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **

            data_shuffler: Data shuffler :py:class:`bob.learn.tensorflow.datashuffler.Base`

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

    def compute_validation(self, step):
        """
        Computes the loss in the validation set

        ** Parameters **
            session: Tensorflow session
            data_shuffler: The data shuffler to be used
            step: Iteration number

        """

        if self.validation_data_shuffler.prefetch:
            l, lr, summary = self.session.run([self.validation_predictor,
                                               self.learning_rate, self.summaries_validation])
        else:
            feed_dict = self.get_feed_dict(self.validation_data_shuffler)
            l, lr, summary = self.session.run([self.validation_predictor,
                                               self.learning_rate, self.summaries_validation],
                                               feed_dict=feed_dict)

        logger.info("Loss VALIDATION set step={0} = {1}".format(step, l))
        self.validation_summary_writter.add_summary(summary, step)               

    def create_general_summary(self, average_loss, output, label):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """
        # Train summary
        tf.summary.scalar('loss', average_loss)
        tf.summary.scalar('lr', self.learning_rate)        

        # Computing accuracy
        correct_prediction = tf.equal(tf.argmax(output, 1), label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)        
        return tf.summary.merge_all()

    def start_thread(self):
        """
        Start pool of threads for pre-fetching

        **Parameters**
          session: Tensorflow session
        """

        threads = []
        for n in range(self.train_data_shuffler.prefetch_threads):
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

            data_ph = self.train_data_shuffler("data", from_queue=False)
            label_ph = self.train_data_shuffler("label", from_queue=False)

            feed_dict = {data_ph: train_data,
                         label_ph: train_labels}

            self.session.run(self.train_data_shuffler.enqueue_op, feed_dict=feed_dict)


