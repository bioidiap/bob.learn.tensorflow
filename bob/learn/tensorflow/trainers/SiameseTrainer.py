#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from ..analyzers import ExperimentAnalizer, SoftmaxAnalizer
from ..network import SequenceNetwork
from .Trainer import Trainer
import os


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
                 architecture,
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
                 prefetch=False,
                 validation_snapshot=100,

                 ## Analizer
                 analizer=ExperimentAnalizer(),

                 model_from_file="",

                 verbosity_level=2):

        super(SiameseTrainer, self).__init__(
            architecture=architecture,
            optimizer=optimizer,
            use_gpu=use_gpu,
            loss=loss,
            temp_dir=temp_dir,

            # Learning rate
            learning_rate=learning_rate,

            ###### training options ##########
            convergence_threshold=convergence_threshold,
            iterations=iterations,
            snapshot=snapshot,
            validation_snapshot=validation_snapshot,
            prefetch=prefetch,

            ## Analizer
            analizer=analizer,

            model_from_file=model_from_file,

            verbosity_level=verbosity_level
        )

        self.between_class_graph_train = None
        self.within_class_graph_train = None

        self.between_class_graph_validation = None
        self.within_class_graph_validation = None

    def bootstrap_placeholders(self, train_data_shuffler, validation_data_shuffler):
        """
        Persist the placeholders
        """

        # Persisting the placeholders
        if self.prefetch:
            batch, batch2, label = train_data_shuffler.get_placeholders_forprefetch()
        else:
            batch, batch2, label = train_data_shuffler.get_placeholders()

        tf.add_to_collection("train_placeholder_data", batch)
        tf.add_to_collection("train_placeholder_data2", batch2)
        tf.add_to_collection("train_placeholder_label", label)

        # Creating validation graph
        if validation_data_shuffler is not None:
            batch, batch2, label = validation_data_shuffler.get_placeholders()
            tf.add_to_collection("validation_placeholder_data", batch)
            tf.add_to_collection("validation_placeholder_data2", batch2)
            tf.add_to_collection("validation_placeholder_label", label)

    def bootstrap_placeholders_fromfile(self, train_data_shuffler, validation_data_shuffler):
        """
        Load placeholders from file
        """

        train_data_shuffler.set_placeholders(tf.get_collection("train_placeholder_data")[0],
                                             tf.get_collection("train_placeholder_data2")[0],
                                             tf.get_collection("train_placeholder_label")[0])

        if validation_data_shuffler is not None:
            train_data_shuffler.set_placeholders(tf.get_collection("validation_placeholder_data")[0],
                                                 tf.get_collection("validation_placeholder_data2")[0],
                                                 tf.get_collection("validation_placeholder_label")[0])

    def compute_graph(self, data_shuffler, prefetch=False, name="", training=True):
        """
        Computes the graph for the trainer.


        ** Parameters **

            data_shuffler: Data shuffler
            prefetch:
            name: Name of the graph
        """

        # Defining place holders
        if prefetch:
            [placeholder_left_data, placeholder_right_data, placeholder_labels] = data_shuffler.get_placeholders_forprefetch(name=name)

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
            [feature_left_batch, feature_right_batch, label_batch] = data_shuffler.get_placeholders(name=name)

        # Creating the siamese graph
        train_left_graph = self.architecture.compute_graph(feature_left_batch, training=training)
        train_right_graph = self.architecture.compute_graph(feature_right_batch, training=training)

        graph, between_class_graph, within_class_graph = self.loss(label_batch,
                                                                   train_left_graph,
                                                                   train_right_graph)

        if training:
            self.between_class_graph_train = between_class_graph
            self.within_class_graph_train = within_class_graph
        else:
            self.between_class_graph_validation = between_class_graph
            self.within_class_graph_validation = within_class_graph

        return graph

    def get_feed_dict(self, data_shuffler):
        """
        Given a data shuffler prepared the dictionary to be injected in the graph

        ** Parameters **
            data_shuffler:

        """

        [batch_left, batch_right, labels] = data_shuffler.get_batch()
        [placeholder_left_data, placeholder_right_data, placeholder_label] = data_shuffler.get_placeholders()

        feed_dict = {placeholder_left_data: batch_left,
                     placeholder_right_data: batch_right,
                     placeholder_label: labels}

        return feed_dict

    def fit(self, step):
        """
        Run one iteration (`forward` and `backward`)

        ** Parameters **
            session: Tensorflow session
            step: Iteration number

        """
        if self.prefetch:
            _, l, bt_class, wt_class, lr, summary = self.session.run([self.optimizer,
                                                    self.training_graph, self.between_class_graph_train, self.within_class_graph_train,
                                                    self.learning_rate, self.summaries_train])
        else:
            feed_dict = self.get_feed_dict(self.train_data_shuffler)
            _, l, bt_class, wt_class, lr, summary = self.session.run([self.optimizer,
                                             self.training_graph, self.between_class_graph_train, self.within_class_graph_train,
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

        if self.validation_summary_writter is None:
            self.validation_summary_writter = tf.train.SummaryWriter(os.path.join(self.temp_dir, 'validation'), self.session.graph)

        self.validation_graph = self.compute_graph(data_shuffler, name="validation", training=False)
        feed_dict = self.get_feed_dict(data_shuffler)
        l, bt_class, wt_class = self.session.run([self.validation_graph,
                                                  self.between_class_graph_validation, self.within_class_graph_validation],
                                                  feed_dict=feed_dict)

        summaries = []
        summaries.append(summary_pb2.Summary.Value(tag="loss", simple_value=float(l)))
        summaries.append(summary_pb2.Summary.Value(tag="between_class_loss", simple_value=float(bt_class)))
        summaries.append(summary_pb2.Summary.Value(tag="within_class_loss", simple_value=float(wt_class)))
        self.validation_summary_writter.add_summary(summary_pb2.Summary(value=summaries), step)
        logger.info("Loss VALIDATION set step={0} = {1}".format(step, l))

    def create_general_summary(self):
        """
        Creates a simple tensorboard summary with the value of the loss and learning rate
        """

        # Train summary
        tf.scalar_summary('loss', self.training_graph, name="train")
        tf.scalar_summary('between_class_loss', self.between_class_graph_train, name="train")
        tf.scalar_summary('within_class_loss', self.within_class_graph_train, name="train")
        tf.scalar_summary('lr', self.learning_rate, name="train")
        return tf.merge_all_summaries()

    def load_and_enqueue(self):
        """
        Injecting data in the place holder queue

        **Parameters**
          session: Tensorflow session
        """

        while not self.thread_pool.should_stop():

            [batch_left, batch_right, labels] = self.train_data_shuffler.get_batch()
            [placeholder_left_data, placeholder_right_data, placeholder_label] = self.train_data_shuffler.get_placeholders()

            feed_dict = {placeholder_left_data: batch_left,
                         placeholder_right_data: batch_right,
                         placeholder_label: labels}

            self.session.run(self.enqueue_op, feed_dict=feed_dict)
