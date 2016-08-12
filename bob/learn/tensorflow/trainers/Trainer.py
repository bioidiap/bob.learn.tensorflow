#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 15:25:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
from ..DataShuffler import DataShuffler
import tensorflow as tf
from ..network import SequenceNetwork
import numpy
from bob.learn.tensorflow.layers import InputLayer


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
        validation_placeholder_data, validation_placeholder_labels = data_shuffler.get_placeholders(name="validation",
                                                                                                    train_dataset=False)

        # Creating the architecture for train and validation
        if not isinstance(self.architecture, SequenceNetwork):
            raise ValueError("The variable `architecture` must be an instance of "
                             "`bob.learn.tensorflow.network.SequenceNetwork`")

        #input_layer = InputLayer(name="input", input_data=train_placeholder_data)

        import ipdb;
        ipdb.set_trace();

        train_graph = self.architecture.compute_graph(train_placeholder_data)

        validation_graph = self.architecture.compute_graph(validation_placeholder_data)

        loss_train = tf.reduce_mean(self.loss(train_graph, train_placeholder_labels))
        loss_validation = tf.reduce_mean(self.loss(validation_graph, validation_placeholder_labels))

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Learning rate
            batch * data_shuffler.train_batch_size,
            data_shuffler.train_data.shape[0],
            self.weight_decay  # Decay step
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_train,
                                                                              global_step=batch)
        train_prediction = tf.nn.softmax(train_graph)
        validation_prediction = tf.nn.softmax(validation_graph)

        print("Initializing !!")
        # Training
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for step in range(self.iterations):

                train_data, train_labels = data_shuffler.get_batch()

                feed_dict = {train_placeholder_data: train_data,
                             train_placeholder_labels: train_labels}

                _, l, lr, _ = session.run([optimizer, loss_train,
                                          learning_rate, train_prediction], feed_dict=feed_dict)

                if step % self.snapshot == 0:
                    validation_data, validation_labels = data_shuffler.get_batch(train_dataset=False)
                    feed_dict = {validation_placeholder_data: validation_data,
                                 validation_placeholder_labels: validation_labels}

                    import ipdb;
                    ipdb.set_trace();

                    l, predictions = session.run([loss_validation, validation_prediction], feed_dict=feed_dict)
                    accuracy = 100. * numpy.sum(numpy.argmax(predictions, 1) == validation_labels) / predictions.shape[0]

                    print "Step {0}. Loss = {1}, Acc Validation={2}".format(step, l, accuracy)

                    #accuracy = util.evaluate_softmax(validation_data, validation_labels, session, validation_prediction,
                    #                                 validation_data_node)
                    #print("Step {0}. Loss = {1}, Lr={2}, Accuracy validation = {3}".format(step, l, lr, accuracy))
                    #sys.stdout.flush()
