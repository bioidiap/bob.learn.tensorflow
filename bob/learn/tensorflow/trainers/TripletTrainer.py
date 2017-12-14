#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import threading
from ..analyzers import ExperimentAnalizer
from .Trainer import Trainer
from ..analyzers import SoftmaxAnalizer
import os
from bob.learn.tensorflow.utils.session import Session
import bob.core
import logging
logger = logging.getLogger("bob.learn")


class TripletTrainer(Trainer):
    """
    Trainer for Triple networks:

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

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

    def __init__(
            self,
            train_data_shuffler,
            validation_data_shuffler=None,
            validate_with_embeddings=False,

            ###### training options ##########
            iterations=5000,
            snapshot=500,
            validation_snapshot=100,
            keep_checkpoint_every_n_hours=2,

            ## Analizer
            analizer=SoftmaxAnalizer(),

            # Temporatu dir
            temp_dir="triplet_cnn",
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
        self.validate_with_embeddings = validate_with_embeddings

        # Analizer
        self.analizer = analizer
        self.global_step = None

        self.session = None

        self.graph = None
        self.validation_graph = None

        self.loss = None

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

    def create_network_from_scratch(
            self,
            graph,
            validation_graph=None,
            optimizer=tf.train.AdamOptimizer(),
            loss=None,

            # Learning rate
            learning_rate=None,
    ):

        self.data_ph = self.train_data_shuffler("data")

        self.graph = graph
        if "anchor" and "positive" and "negative" not in self.graph:
            raise ValueError(
                "`graph` should be a dictionary with two elements (`anchor`, `positive` and `negative`)"
            )

        self.loss = loss
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate

        self.global_step = tf.train.get_or_create_global_step()

        # Saving all the variables
        self.saver = tf.train.Saver(var_list=tf.global_variables())

        tf.add_to_collection("global_step", self.global_step)

        # Saving the pointers to the graph
        tf.add_to_collection("graph_anchor", self.graph['anchor'])
        tf.add_to_collection("graph_positive", self.graph['positive'])
        tf.add_to_collection("graph_negative", self.graph['negative'])

        # Saving pointers to the loss
        tf.add_to_collection("loss", self.loss['loss'])
        tf.add_to_collection("between_class_loss", self.loss['between_class'])
        tf.add_to_collection("within_class_loss", self.loss['within_class'])

        # Saving the pointers to the placeholders
        tf.add_to_collection("data_ph_anchor", self.data_ph['anchor'])
        tf.add_to_collection("data_ph_positive", self.data_ph['positive'])
        tf.add_to_collection("data_ph_negative", self.data_ph['negative'])

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(
            self.loss['loss'], global_step=self.global_step)
        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        self.summaries_train = self.create_general_summary()
        tf.add_to_collection("summaries_train", self.summaries_train)

        # Creating the variables
        tf.global_variables_initializer().run(session=self.session)

    def create_network_from_file(self, file_name, clear_devices=True):

        self.load_checkpoint(file_name, clear_devices=clear_devices)

        # Loading the graph from the graph pointers
        self.graph = dict()
        self.graph['anchor'] = tf.get_collection("graph_anchor")[0]
        self.graph['positive'] = tf.get_collection("graph_positive")[0]
        self.graph['negative'] = tf.get_collection("graph_negative")[0]

        # Loading the placeholders from the pointers
        self.data_ph = dict()
        self.data_ph['anchor'] = tf.get_collection("data_ph_anchor")[0]
        self.data_ph['positive'] = tf.get_collection("data_ph_positive")[0]
        self.data_ph['negative'] = tf.get_collection("data_ph_negative")[0]

        # Loading loss from the pointers
        self.loss = dict()
        self.loss['loss'] = tf.get_collection("loss")[0]
        self.loss['between_class'] = tf.get_collection("between_class_loss")[0]
        self.loss['within_class'] = tf.get_collection("within_class_loss")[0]

        # Loading other elements
        self.optimizer = tf.get_collection("optimizer")[0]
        self.learning_rate = tf.get_collection("learning_rate")[0]
        self.summaries_train = tf.get_collection("summaries_train")[0]
        self.global_step = tf.get_collection("global_step")[0]
        self.from_scratch = False

    def get_feed_dict(self, data_shuffler):

        [batch_anchor, batch_positive,
         batch_negative] = data_shuffler.get_batch()
        feed_dict = {
            self.data_ph['anchor']: batch_anchor,
            self.data_ph['positive']: batch_positive,
            self.data_ph['negative']: batch_negative
        }

        return feed_dict

    def fit(self, step):
        feed_dict = self.get_feed_dict(self.train_data_shuffler)
        _, l, bt_class, wt_class, lr, summary = self.session.run(
            [
                self.optimizer, self.loss['loss'], self.loss['between_class'],
                self.loss['within_class'], self.learning_rate,
                self.summaries_train
            ],
            feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    def create_general_summary(self):

        # Train summary
        tf.summary.scalar('loss', self.loss['loss'])
        tf.summary.scalar('between_class_loss', self.loss['between_class'])
        tf.summary.scalar('within_class_loss', self.loss['within_class'])
        tf.summary.scalar('lr', self.learning_rate)
        return tf.summary.merge_all()

    def load_and_enqueue(self):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session

        """
        while not self.thread_pool.should_stop():
            [train_data_anchor, train_data_positive,
             train_data_negative] = self.train_data_shuffler.get_batch()

            data_ph = dict()
            data_ph['anchor'] = self.train_data_shuffler(
                "data", from_queue=False)['anchor']
            data_ph['positive'] = self.train_data_shuffler(
                "data", from_queue=False)['positive']
            data_ph['negative'] = self.train_data_shuffler(
                "data", from_queue=False)['negative']

            feed_dict = {
                data_ph['anchor']: train_data_anchor,
                data_ph['positive']: train_data_positive,
                data_ph['negative']: train_data_negative
            }

            self.session.run(
                self.train_data_shuffler.enqueue_op, feed_dict=feed_dict)
