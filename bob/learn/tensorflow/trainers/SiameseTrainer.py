#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
from ..analyzers import Analizer
from ..network import SequenceNetwork


class SiameseTrainer(object):

    def __init__(self,

                 architecture=None,
                 use_gpu=False,
                 loss=None,

                 ###### training options ##########
                 convergence_threshold = 0.01,
                 iterations=5000,
                 base_lr=0.001,
                 momentum=0.9,
                 weight_decay=0.95,

                 # The learning rate policy
                 snapshot=100):

        self.loss = loss
        self.loss_instance = None
        self.optimizer = None

        self.architecture = architecture
        self.use_gpu = use_gpu

        self.iterations = iterations
        self.snapshot = snapshot
        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.convergence_threshold = convergence_threshold

    def train(self, data_shuffler):
        """
        Do the loop forward --> backward --|
                      ^--------------------|
        """

        train_placeholder_left_data, train_placeholder_labels = data_shuffler.get_placeholders(name="train_left")
        train_placeholder_right_data, _ = data_shuffler.get_placeholders(name="train_right")
        feature_placeholder, _ = data_shuffler.get_placeholders(name="feature", train_dataset=False)

        #validation_placeholder_data, validation_placeholder_labels = data_shuffler.get_placeholders(name="validation",
        #                                                                                            train_dataset=False)

        # Creating the architecture for train and validation
        if not isinstance(self.architecture, SequenceNetwork):
            raise ValueError("The variable `architecture` must be an instance of "
                             "`bob.learn.tensorflow.network.SequenceNetwork`")

        train_left_graph = self.architecture.compute_graph(train_placeholder_left_data)
        train_right_graph = self.architecture.compute_graph(train_placeholder_right_data)
        feature_graph = self.architecture.compute_graph(feature_placeholder, cut=True)

        loss_train, between_class, within_class = self.loss(train_placeholder_labels,
                                                            train_left_graph,
                                                            train_right_graph,
                                                            0.2)

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Learning rate
            batch * data_shuffler.train_batch_size,
            data_shuffler.train_data.shape[0],
            self.weight_decay  # Decay step
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_train,
                                                                              global_step=batch)

        #train_prediction = tf.nn.softmax(train_graph)
        #validation_prediction = tf.nn.softmax(validation_graph)


        print("Initializing !!")
        # Training
        with tf.Session() as session:
            analizer = Analizer(data_shuffler, feature_graph, feature_placeholder, session)

            train_writer = tf.train.SummaryWriter('./LOGS/train',
                                                  session.graph)

            # Tensorboard data
            tf.scalar_summary('loss', loss_train)
            tf.scalar_summary('between_class', between_class)
            tf.scalar_summary('within_class', within_class)
            tf.scalar_summary('lr', learning_rate)
            merged = tf.merge_all_summaries()

            tf.initialize_all_variables().run()
            for step in range(self.iterations):

                batch_left, batch_right, labels = data_shuffler.get_pair()

                feed_dict = {train_placeholder_left_data: batch_left,
                             train_placeholder_right_data: batch_right,
                             train_placeholder_labels: labels}

                _, l, lr, summary = session.run([optimizer, loss_train, learning_rate, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)

                if step % self.snapshot == 0:
                    analizer()

            train_writer.close()