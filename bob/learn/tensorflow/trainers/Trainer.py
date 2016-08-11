#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
from ..DataShuffler import DataShuffler
import tensorflow as tf
from ..network import SequenceNetwork


class Trainer(object):

    def __init__(self,

                 architecture=None,
                 use_gpu=False,
                 loss=None,

                 ###### training options ##########
                 convergence_threshold = 0.01,
                 iterations=5000,
                 base_lr=0.00001,
                 momentum=0.9,
                 weight_decay=0.0005,

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

        train_placeholder_data, train_placeholder_labels = data_shuffler.get_placeholders(name="train")
        validation_placeholder_data, validation_placeholder_labels = data_shuffler.get_placeholders(name="validation")

        # Creating the architecture for train and validation
        if not isinstance(self.architecture, SequenceNetwork):
            raise ValueError("The variable `architecture` must be an instance of "
                             "`bob.learn.tensorflow.network.SequenceNetwork`")

        train_graph = self.architecture.compute_graph(train_placeholder_data)
        loss_instance = tf.reduce_mean(self.loss(train_graph))

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Learning rate
            batch * self.train_batch_size,
            data_shuffler.train_data.shape[0],
            self.weight_decay  # Decay step
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_instance,
                                                                              global_step=batch)
        train_prediction = tf.nn.softmax(train_graph)

        print("Initializing !!")
        # Training
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for step in range(self.iterations):

                train_data, train_labels = data_shuffler.get_batch(self.train_batch_size)

                feed_dict = {train_placeholder_data: train_data,
                             train_placeholder_labels: train_labels}

                _, l, lr, predictions = session.run([self.optimizer, self.loss_instance,
                                                     self.learning_rate, train_prediction], feed_dict=feed_dict)

                if step % self.snapshot == 0:
                    validation_data, validation_labels = data_shuffler.get_batch(data_shuffler.validation_data.shape[0],
                                                                                 train_dataset=False)
                    feed_dict = {validation_placeholder_data: validation_data,
                                 validation_placeholder_labels: validation_labels}

                    l, predictions = session.run([self.loss_instance, train_prediction], feed_dict=feed_dict)
                    print("Step {0}. Loss = {1}, Lr={2}".format(step, l, predictions))

                    #accuracy = util.evaluate_softmax(validation_data, validation_labels, session, validation_prediction,
                    #                                 validation_data_node)
                    #print("Step {0}. Loss = {1}, Lr={2}, Accuracy validation = {3}".format(step, l, lr, accuracy))
                    #sys.stdout.flush()
