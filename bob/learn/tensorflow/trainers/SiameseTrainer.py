#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from .Trainer import Trainer
from ..analyzers import SoftmaxAnalizer
from .learning_rate import constant
import os
import logging
from bob.learn.tensorflow.utils.session import Session
import bob.core
logger = logging.getLogger("bob.learn")


class SiameseTrainer(Trainer):
    """
    Trainer for siamese networks:
     
    Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to
    face verification." 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05). Vol. 1. IEEE, 2005.
    

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
        self.summaries_validation = None

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
        self.label_ph = self.train_data_shuffler("label")

        self.graph = graph
        if "left" and "right" not in self.graph:
            raise ValueError("`graph` should be a dictionary with two elements (`left`and `right`)")

        self.loss = loss
        self.predictor = self.loss(self.label_ph,
                                   self.graph["left"],
                                   self.graph["right"])
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate

        # TODO: find an elegant way to provide this as a parameter of the trainer
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Saving all the variables
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        tf.add_to_collection("global_step", self.global_step)

        # Saving the pointers to the graph
        tf.add_to_collection("graph_left", self.graph['left'])
        tf.add_to_collection("graph_right", self.graph['right'])

        # Saving pointers to the loss
        tf.add_to_collection("predictor_loss", self.predictor['loss'])
        tf.add_to_collection("predictor_between_class_loss", self.predictor['between_class'])
        tf.add_to_collection("predictor_within_class_loss", self.predictor['within_class'])

        # Saving the pointers to the placeholders
        tf.add_to_collection("data_ph_left", self.data_ph['left'])
        tf.add_to_collection("data_ph_right", self.data_ph['right'])
        tf.add_to_collection("label_ph", self.label_ph)

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.predictor['loss'], global_step=self.global_step)
        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        self.summaries_train = self.create_general_summary()
        tf.add_to_collection("summaries_train", self.summaries_train)

        self.summaries_validation = self.create_general_summary()
        tf.add_to_collection("summaries_validation", self.summaries_validation)

        # Creating the variables
        tf.global_variables_initializer().run(session=self.session)

    def create_network_from_file(self, model_from_file, clear_devices=True):

        #saver = self.architecture.load(self.model_from_file, clear_devices=False)
        self.saver = tf.train.import_meta_graph(model_from_file + ".meta", clear_devices=clear_devices)
        self.saver.restore(self.session, model_from_file)

        # Loading the graph from the graph pointers
        self.graph = dict()
        self.graph['left'] = tf.get_collection("graph_left")[0]
        self.graph['right'] = tf.get_collection("graph_right")[0]

        # Loading the placeholders from the pointers
        self.data_ph = dict()
        self.data_ph['left'] = tf.get_collection("data_ph_left")[0]
        self.data_ph['right'] = tf.get_collection("data_ph_right")[0]
        self.label_ph = tf.get_collection("label_ph")[0]

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

        [batch_left, batch_right, labels] = data_shuffler.get_batch()

        feed_dict = {self.data_ph['left']: batch_left,
                     self.data_ph['right']: batch_right,
                     self.label_ph: labels}

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

    def compute_validation(self, data_shuffler, step):
        """
        Computes the loss in the validation set

        ** Parameters **
            session: Tensorflow session
            data_shuffler: The data shuffler to be used
            step: Iteration number

        """
        # Opening a new session for validation
        feed_dict = self.get_feed_dict(data_shuffler)

        l, summary = self.session.run([self.predictor, self.summaries_validation], feed_dict=feed_dict)
        self.validation_summary_writter.add_summary(summary, step)

        #summaries = [summary_pb2.Summary.Value(tag="loss", simple_value=float(l))]
        #self.validation_summary_writter.add_summary(summary_pb2.Summary(value=summaries), step)
        logger.info("Loss VALIDATION set step={0} = {1}".format(step, l))

    def load_and_enqueue(self):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session

        """
        while not self.thread_pool.should_stop():
            [train_data_left, train_data_right, train_labels] = self.train_data_shuffler.get_batch()

            data_ph = dict()
            data_ph['left'] = self.train_data_shuffler("data", from_queue=False)['left']
            data_ph['right'] = self.train_data_shuffler("data", from_queue=False)['right']
            label_ph = self.train_data_shuffler("label", from_queue=False)

            feed_dict = {data_ph['left']: train_data_left,
                         data_ph['right']: train_data_right,
                         label_ph: train_labels}

            self.session.run(self.train_data_shuffler.enqueue_op, feed_dict=feed_dict)
