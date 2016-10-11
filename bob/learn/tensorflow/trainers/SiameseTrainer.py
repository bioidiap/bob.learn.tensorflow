#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import threading
from ..analyzers import ExperimentAnalizer, SoftmaxAnalizer
from ..network import SequenceNetwork
import bob.io.base
from .Trainer import Trainer
import os
import sys


class SiameseTrainer(Trainer):

    """
    Trainer for siamese networks.

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

        super(SiameseTrainer, self).__init__(
            architecture=architecture,
            optimizer=optimizer,
            use_gpu=use_gpu,
            loss=loss,
            temp_dir=temp_dir,

            # Learning rate
            base_learning_rate=base_learning_rate,
            weight_decay=weight_decay,

            ###### training options ##########
            convergence_threshold=convergence_threshold,
            iterations=iterations,
            snapshot=snapshot,
            prefetch=prefetch,

            ## Analizer
            analizer=analizer,

            verbosity_level=verbosity_level
        )

        self.between_class_graph = None
        self.within_class_graph = None

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
            placeholder_left_data, placeholder_right_data, placeholder_labels = data_shuffler.get_placeholders_pair_forprefetch(name="train")

            # Creating two graphs
            #placeholder_left_data, placeholder_labels = data_shuffler. \
            #    get_placeholders_forprefetch(name="train_left")
            #placeholder_right_data, _ = data_shuffler.get_placeholders(name="train_right")
            feature_left_batch, feature_right_batch, label_batch = data_shuffler.get_placeholders_pair(name="train_")

            # Defining a placeholder queue for prefetching
            queue = tf.FIFOQueue(capacity=100,
                                 dtypes=[tf.float32, tf.float32, tf.int64],
                                 shapes=[placeholder_left_data.get_shape().as_list()[1:],
                                         placeholder_right_data.get_shape().as_list()[1:],
                                         []])

            # Fetching the place holders from the queue
            self.enqueue_op = queue.enqueue_many([placeholder_left_data, placeholder_right_data, placeholder_labels])
            feature_left_batch, feature_right_batch, label_batch = queue.dequeue_many(data_shuffler.batch_size)

            # Creating the architecture for train and validation
            if not isinstance(self.architecture, SequenceNetwork):
                raise ValueError("The variable `architecture` must be an instance of "
                                 "`bob.learn.tensorflow.network.SequenceNetwork`")
        else:
            feature_left_batch, feature_right_batch, label_batch = data_shuffler.get_placeholders_pair(name="train_")
            #feature_left_batch, label_batch = data_shuffler.get_placeholders(name="train_left")
            #feature_right_batch, _ = data_shuffler.get_placeholders(name="train_right")

        # Creating the siamese graph
        train_left_graph = self.architecture.compute_graph(feature_left_batch)
        train_right_graph = self.architecture.compute_graph(feature_right_batch)

        graph, between_class_graph, within_class_graph = self.loss(label_batch,
                                                                   train_left_graph,
                                                                   train_right_graph)

        self.between_class_graph = between_class_graph
        self.within_class_graph = within_class_graph

        return graph

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """

        batch_left, batch_right, labels = data_shuffler.get_pair()
        placeholder_left_data, placeholder_right_data, placeholder_label = data_shuffler.get_placeholders_pair(name="train")

        feed_dict = {placeholder_left_data: batch_left,
                     placeholder_right_data: batch_right,
                     placeholder_label: labels}

        return feed_dict

    def fit(self, session, step):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """
        if self.prefetch:
            _, l, bt_class, wt_class, lr, summary = session.run([self.optimizer,
                                             self.training_graph, self.between_class_graph, self.within_class_graph,
                                             self.learning_rate, self.summaries_train])
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            _, l, bt_class, wt_class, lr, summary = session.run([self.optimizer,
                                             self.training_graph, self.between_class_graph, self.within_class_graph,
                                             self.learning_rate, self.summaries_train], feed_dict=feed_dict)

        logger.info("Loss training set step={0} = {1}".format(step, l))
        self.train_summary_writter.add_summary(summary, step)

    def compute_validation(self, session, data_shuffler, step):
        """
        Computes the loss in the validation set

        ** Parameters **
            session: Tensorflow session
            data_shuffler: The data shuffler to be used
            step: Iteration number

        """

        if self.validation_summary_writter is None:
            self.validation_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'validation'), session.graph)

        self.validation_graph = self.compute_graph(data_shuffler, name="validation")
        feed_dict = self.get_feed_dict(data_shuffler)
        l = session.run(self.validation_graph, feed_dict=feed_dict)

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
        tf.scalar_summary('between_class_loss', self.between_class_graph, name="train")
        tf.scalar_summary('within_class_loss', self.within_class_graph, name="train")
        tf.scalar_summary('lr', self.learning_rate, name="train")
        return tf.merge_all_summaries()

    def load_and_enqueue(self, session):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session
        """

        while not self.thread_pool.should_stop():

            batch_left, batch_right, labels = self.train_data_shuffler.get_pair()
            placeholder_left_data, placeholder_right_data, placeholder_label = self.train_data_shuffler.get_placeholders_pair()

            feed_dict = {placeholder_left_data: batch_left,
                         placeholder_right_data: batch_right,
                         placeholder_label: labels}

            session.run(self.enqueue_op, feed_dict=feed_dict)
