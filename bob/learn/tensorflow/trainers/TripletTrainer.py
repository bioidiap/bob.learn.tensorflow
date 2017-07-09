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

    def __init__(self,
                 train_data_shuffler,

                 ###### training options ##########
                 iterations=5000,
                 snapshot=500,
                 validation_snapshot=100,

                 ## Analizer
                 analizer=SoftmaxAnalizer(),

                 # Temporatu dir
                 temp_dir="siamese_cnn",

                 verbosity_level=2
                 ):

        super(TripletTrainer, self).__init__(
            train_data_shuffler,

            ###### training options ##########
            iterations=5000,
            snapshot=500,
            validation_snapshot=100,

            ## Analizer
            analizer=SoftmaxAnalizer(),

            # Temporatu dir
            temp_dir="siamese_cnn",

            verbosity_level=2
        )

        self.train_data_shuffler = train_data_shuffler
        self.temp_dir = temp_dir

        self.iterations = iterations
        self.snapshot = snapshot
        self.validation_snapshot = validation_snapshot

        # Training variables used in the fit
        self.summaries_train = None
        self.train_summary_writter = None

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

        bob.core.log.set_verbosity_level(logger, verbosity_level)

    def create_network_from_scratch(self,
                                    graph,
                                    optimizer=tf.train.AdamOptimizer(),
                                    loss=None,

                                    # Learning rate
                                    learning_rate=None,
                                    ):

        self.data_ph = self.train_data_shuffler("data")

        self.graph = graph
        if "anchor" and "positive" and "negative" not in self.graph:
            raise ValueError("`graph` should be a dictionary with two elements (`anchor`, `positive` and `negative`)")

        self.loss = loss
        self.predictor = self.loss(self.graph["anchor"],
                                   self.graph["positive"],
                                   self.graph["negative"])
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate

        # TODO: find an elegant way to provide this as a parameter of the trainer
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Saving all the variables
        self.saver = tf.train.Saver(var_list=tf.global_variables())

        tf.add_to_collection("global_step", self.global_step)

        # Saving the pointers to the graph
        tf.add_to_collection("graph_anchor", self.graph['anchor'])
        tf.add_to_collection("graph_positive", self.graph['positive'])
        tf.add_to_collection("graph_negative", self.graph['negative'])

        # Saving pointers to the loss
        tf.add_to_collection("predictor_loss", self.predictor['loss'])
        tf.add_to_collection("predictor_between_class_loss", self.predictor['between_class'])
        tf.add_to_collection("predictor_within_class_loss", self.predictor['within_class'])

        # Saving the pointers to the placeholders
        tf.add_to_collection("data_ph_anchor", self.data_ph['anchor'])
        tf.add_to_collection("data_ph_positive", self.data_ph['positive'])
        tf.add_to_collection("data_ph_negative", self.data_ph['negative'])

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.predictor['loss'], global_step=self.global_step)
        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        self.summaries_train = self.create_general_summary()
        tf.add_to_collection("summaries_train", self.summaries_train)

        # Creating the variables
        tf.global_variables_initializer().run(session=self.session)

    def create_network_from_file(self, model_from_file, clear_devices=True):

        #saver = self.architecture.load(self.model_from_file, clear_devices=False)
        self.saver = tf.train.import_meta_graph(model_from_file + ".meta", clear_devices=clear_devices)
        self.saver.restore(self.session, model_from_file)

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
        self.predictor = dict()
        self.predictor['loss'] = tf.get_collection("predictor_loss")[0]
        self.predictor['between_class'] = tf.get_collection("predictor_between_class_loss")[0]
        self.predictor['within_class'] = tf.get_collection("predictor_within_class_loss")[0]

        # Loading other elements
        self.optimizer = tf.get_collection("optimizer")[0]
        self.learning_rate = tf.get_collection("learning_rate")[0]
        self.summaries_train = tf.get_collection("summaries_train")[0]
        self.global_step = tf.get_collection("global_step")[0]
        self.from_scratch = False

    def get_feed_dict(self, data_shuffler):

        [batch_anchor, batch_positive, batch_negative] = data_shuffler.get_batch()
        feed_dict = {self.data_ph['anchor']: batch_anchor,
                     self.data_ph['positive']: batch_positive,
                     self.data_ph['negative']: batch_negative}

        return feed_dict

    def fit(self, step):
        feed_dict = self.get_feed_dict(self.train_data_shuffler)
        _, l, bt_class, wt_class, lr, summary = self.session.run([
                                                self.optimizer,
                                                self.predictor['loss'], self.predictor['between_class'],
                                                self.predictor['within_class'],
                                                self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    def create_general_summary(self):

        # Train summary
        tf.summary.scalar('loss', self.predictor['loss'])
        tf.summary.scalar('between_class_loss', self.predictor['between_class'])
        tf.summary.scalar('within_class_loss', self.predictor['within_class'])
        tf.summary.scalar('lr', self.learning_rate)
        return tf.summary.merge_all()

