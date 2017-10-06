#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 13 Sep 2017 13:22 CEST

import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.framework import ops

import logging
logger = logging.getLogger("bob.learn")


def lstm(inputs, lstm_cell_size, lstm_fn=tf.contrib.rnn.BasicLSTMCell, num_time_steps=20,
         output_activation_size=10, batch_size=10, scope='rnn',
         weights_initializer=tf.random_normal, activation=tf.nn.relu,
         name=None, reuse=None, dropout=False, input_dropout=1.0, output_dropout=1.0):
    """
    """
    return LSTM(lstm_cell_size=lstm_cell_size,
                num_time_steps=num_time_steps,
                batch_size=batch_size,
                lstm_fn=lstm_fn,
                scope=scope,
                output_activation_size=output_activation_size,
                weights_initializer=weights_initializer,
                activation=activation,
                dropout=dropout,
                input_dropout=input_dropout,
                output_dropout=output_dropout,
                name=name,
                reuse=reuse)(inputs)


class LSTM(base.Layer):
    """
    Basic LSTM layer in the format of tf-slim
    """

    def __init__(self, lstm_cell_size,
                 num_time_steps=20,
                 batch_size=10,
                 lstm_fn=tf.contrib.rnn.BasicLSTMCell,
                 output_activation_size=10,
                 scope='rnn',
                 weights_initializer=tf.random_normal,
                 activation=tf.nn.relu,
                 name=None,
                 reuse=None,
                 dropout=False,
                 input_dropout=1.0,
                 output_dropout=1.0,
                 **kwargs):
        """
        :param lstm_cell_size [int]: size of the LSTM cell, i.e., the length of the output form each cell
        :param batch_size [int]: input data batch size
        :param num_time_steps [int]: the number of time steps of the input, i.e.,
        the number of LSTM cells in one layer
        """
        super(LSTM, self).__init__(name=name, trainable=False, **kwargs)

        self.lstm_cell_size = lstm_cell_size
        self.lstm = lstm_fn(self.lstm_cell_size, activation=activation, reuse=reuse, state_is_tuple=True, **kwargs)

        # print("self.lstm_cell_size: ", self.lstm_cell_size)
        if dropout:
            self.lstm = tf.nn.rnn_cell.DropoutWrapper(self.lstm, input_keep_prob=input_dropout,
                                                      output_keep_prob=output_dropout)
        self.batch_size = batch_size
        self.num_time_steps = num_time_steps
        # print("self.num_time_steps:", self.num_time_steps)
        self.scope = scope

        hidden_state = tf.zeros([self.batch_size, self.lstm_cell_size])
        current_state = tf.zeros([self.batch_size, self.lstm_cell_size])
        self.states = hidden_state, current_state

        self.sequence_length = None
        self.output_activation_size = output_activation_size

        # Define weights
        # self.output_activation_weights = {
        #     'out': tf.Variable(weights_initializer([self.lstm_cell_size, self.output_activation_size]))
        # }
        # self.output_activation_biases = {
        #     'out': tf.Variable(weights_initializer([self.output_activation_size]))
        # }

    def __call__(self, inputs):
        """
        :param inputs: The input is expected to have shape (batch_size, num_time_steps, input_vector_size).
        """
        # shape inputs correctly
        inputs = ops.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        logger.info("LSTM: the shape of the inputs: {0}".format(shape))

        input_time_steps = shape[1]  # second dimension must be the number of time steps in LSTM

        if len(shape) == 4:  # when inputs shape is 4, the last dimension must be 1
            if shape[-1] == 1:  # we accept last dimension to be 1, then we just reshape it
                inputs = tf.reshape(inputs, shape=(-1, shape[1], shape[2]))
                logger.info("LSTM: after reshape, the shape of the inputs: {0}".format(inputs.get_shape().as_list()))
            else:
                raise ValueError('The shape of input must be either (batch_size, num_time_steps, input_vector_size) or '
                                 '(batch_size, num_time_steps, input_vector_size, 1), but it is {}'.format(shape))

        if input_time_steps % self.num_time_steps:
            raise ValueError('number of rows in one batch of input ({}) should be '
                             'the same as the num_time_steps of LSTM ({})'
                             .format(input_time_steps, self.num_time_steps))

        # convert inputs into the num_time_steps list of the inputs each of shape (batch_size, input_vector_size)
        list_inputs = tf.unstack(inputs, self.num_time_steps, 1)

        # run LSTM training on the batch of inputs
        # return the output (a list of self.num_time_steps outputs each of size input_vector_size)
        # and remember the final states
        outputs, self.states = tf.contrib.rnn.static_rnn(self.lstm,
                                                         inputs=list_inputs,
                                                         initial_state=self.states,
                                                         dtype=tf.float32,
                                                         scope=self.scope)

        # consider the output of the last cell
        return outputs[-1]
        # return tf.matmul(outputs[-1], self.output_activation_weights['out']) + self.output_activation_biases['out']

