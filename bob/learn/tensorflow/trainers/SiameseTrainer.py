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
    Trainer for siamese networks.

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
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """
        [batch_left, batch_right, labels] = data_shuffler.get_batch()

        feed_dict = {self.data_ph['left']: batch_left,
                     self.data_ph['right']: batch_right,
                     self.label_ph: labels}

        return feed_dict

    def fit(self, step):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """
        feed_dict = self.get_feed_dict(self.train_data_shuffler)
        _, l, bt_class, wt_class, lr, summary = self.session.run([
                                                self.optimizer,
                                                self.predictor['loss'], self.predictor['between_class'],
                                                self.predictor['within_class'],
                                                self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    def create_general_summary(self):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """

        # Train summary
        tf.summary.scalar('loss', self.predictor['loss'])
        tf.summary.scalar('between_class_loss', self.predictor['between_class'])
        tf.summary.scalar('within_class_loss', self.predictor['within_class'])
        tf.summary.scalar('lr', self.learning_rate)
        return tf.summary.merge_all()
