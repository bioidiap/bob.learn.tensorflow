#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavle Korshunov <pavel.korshunov@idiap.ch>
# @date: Fri 15 Sep 2017 13:22 CEST

"""
LSTM network architecture.
"""

from bob.learn.tensorflow.layers import lstm
import tensorflow as tf

import bob.core
logger = bob.core.log.setup("bob.project.savi")

slim = tf.contrib.slim

class RegularizedLoss(object):
    """
    Mean softmax loss with regularization
    """

    def __init__(self, name="reg_loss", regularizing_coeff=0.1):
        self.name = name
        self.regularizing_coeff = regularizing_coeff
        tv = tf.trainable_variables()
        self.regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv
                                                  if 'basic_lstm_cell/kernel' in v.name or 'weights' in v.name])
        # for v in tv:
        #     if 'basic_lstm_cell/kernel' in v.name or 'weights' in v.name:
        #         print("regularizing:", v.name)
        #

    def __call__(self, graph, label):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=graph, labels=label), name=self.name)
        # loss = tf.reduce_sum(tf.pow(tf.nn.softmax(graph) - tf.contrib.layers.one_hot_encoding(label, 2), 2))
        # return loss
        return tf.reduce_mean(loss + self.regularizing_coeff * self.regularization_cost)


def simple_lstm_network(train_data_shuffler, lstm_cell_size=64, batch_size=10,
                        num_time_steps=28, num_classes=10, seed=10, reuse=False,
                        dropout=False, input_dropout=1.0, output_dropout=1.0):

    if isinstance(train_data_shuffler, tf.Tensor):
        inputs = train_data_shuffler
    else:
        inputs = train_data_shuffler("data", from_queue=False)

    initializer = tf.contrib.layers.xavier_initializer(seed=seed)

    # Creating an LSTM network
    # graph = tf.contrib.layers.dropout(inputs, keep_prob=0.5, is_training=(not reuse), scope="input_dropout")

    graph = lstm(inputs, lstm_cell_size, num_time_steps=num_time_steps, batch_size=batch_size,
                 output_activation_size=num_classes, scope='lstm', name='sync_cell',
                 weights_initializer=initializer, activation=tf.nn.sigmoid, reuse=reuse,
                 dropout=dropout, input_dropout=input_dropout, output_dropout=output_dropout)

    # graph = tf.contrib.layers.dropout(graph, keep_prob=0.5, is_training=(not reuse), scope="lstm_dropout")

    # graph = tf.layers.batch_normalization(graph, reuse=reuse)
    # graph = tf.contrib.layers.batch_norm(graph, trainable=(not reuse), reuse=reuse, scope='batch_norm')
    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    regularizer = None
    # fully connect the LSTM output to the classes
    graph = slim.fully_connected(graph, num_classes, activation_fn=None, scope='fc1',
                                 weights_initializer=initializer, weights_regularizer=regularizer, reuse=reuse)

    return graph
