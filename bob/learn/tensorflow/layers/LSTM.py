#!/usr/bin/env python

import tensorflow as tf

from tensorflow.python.layers import base

def lstm(inputs, n_hidden, name=None):
    """
    """
    return LSTM(n_hidden=n_hidden, name=name)(inputs)


class LSTM(base.Layer):
    """
    """

    def __init__(self,
                 n_hidden,
                 name=None,
                 **kwargs):
        super(LSTM, self).__init__(name=name,
                                   **kwargs)
        self.n_hidden = n_hidden
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden,
                                                      forget_bias=1.0)

    def call(self, inputs, training=False):
        """
        """

        outputs, states = tf.nn.static_rnn(self.lstm_cell,
                                           inputs,
                                           dtype=tf.float32)

        return outputs[-1]
