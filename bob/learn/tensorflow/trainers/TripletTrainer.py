#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import threading
from ..analyzers import ExperimentAnalizer
from ..network import SequenceNetwork
from .Trainer import Trainer
from ..analyzers import SoftmaxAnalizer
import os
from bob.learn.tensorflow.utils.session import Session
import bob.core
import logging
logger = logging.getLogger("bob.learn")


class TripletTrainer(Trainer):
    """
    Trainer for Triple networks.

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

        tf.add_to_collection("graph", self.graph)
        tf.add_to_collection("predictor", self.predictor)

        tf.add_to_collection("data_ph", self.data_ph)

        # Preparing the optimizer
        self.optimizer_class._learning_rate = self.learning_rate
        self.optimizer = self.optimizer_class.minimize(self.predictor[0], global_step=self.global_step)
        tf.add_to_collection("optimizer", self.optimizer)
        tf.add_to_collection("learning_rate", self.learning_rate)

        self.summaries_train = self.create_general_summary()
        tf.add_to_collection("summaries_train", self.summaries_train)

        # Creating the variables
        tf.global_variables_initializer().run(session=self.session)

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """

        [batch_anchor, batch_positive, batch_negative] = data_shuffler.get_batch()
        feed_dict = {self.data_ph['anchor']: batch_anchor,
                     self.data_ph['positive']: batch_positive,
                     self.data_ph['negative']: batch_negative}

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
                                                self.predictor[0], self.predictor[1],
                                                self.predictor[2],
                                                self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)


    def create_general_summary(self):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """

        # Train summary
        tf.summary.scalar('loss', self.predictor[0])
        tf.summary.scalar('between_class_loss', self.predictor[1])
        tf.summary.scalar('within_class_loss', self.predictor[2])
        tf.summary.scalar('lr', self.learning_rate)
        return tf.summary.merge_all()

