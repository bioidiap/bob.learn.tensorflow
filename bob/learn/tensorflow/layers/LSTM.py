#!/usr/bin/env python

import tensorflow as tf

from tensorflow.python.layers import base

# def lstm(inputs, n_hidden, name=None):
#     """
#     """
#     return LSTM(n_hidden=n_hidden, name=name)(inputs)

def rnn(inputs, n_hidden, cell_fn, cell_args, name=None):
    """
    """
    return RNN(n_hidden=n_hidden,
               cell_fn = cell_fn,
               cell_args = cell_args,
               name=name)(inputs)


class RNN(base.Layer):
    """
    Inspired from tensorlayer/tensorlayer/layers.py
    """

    def __init__(self, n_hidden,
                 cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
                 cell_args = { "forget_bias": 1.0, },
                 name=None,
                 **kwargs):
        """
        """
        super(RNN, self).__init__(name=name, **kwargs)

        self.n_hidden = n_hidden
        self.cell = cell_fn(num_units = self.n_hidden, **kwargs)

    def call(self, inputs, training=False):
        """
        """
        outputs, states = tf.nn.static_rnn(self.cell,
                                           inputs,
                                           dtype=tf.float32)

        # Compare to tensorlayer, it is as if return_last = True
        return outputs[-1]
