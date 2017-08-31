#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.layers import base

def rnn(inputs, n_hidden,
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
        cell_args = { "forget_bias": 1.0, },
        name = None):
    """
    """
    return RNN(n_hidden = n_hidden,
               cell_fn = cell_fn,
               cell_args = cell_args,
               name = name)(inputs)

def rnn3d(inputs, n_hidden,
          cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
          cell_args = { "forget_bias": 1.0, },
          name = None):
    """
    Expect input of size (-1, n_sequence, n_features)

    For instance (batch_size, n_sequence, n_features)
    """
    return RNN3D(n_hidden = n_hidden,
                 cell_fn = cell_fn,
                 cell_args = cell_args,
                 name = name)(inputs)


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

class RNN3D(base.Layer):
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
        super(RNN3D, self).__init__(name=name, **kwargs)

        self.n_hidden = n_hidden
        self.cell = cell_fn(num_units = self.n_hidden, **kwargs)
        self.n_steps = -1
        self.n_features = -1

    def call(self, inputs, training=False):
        """
        """
        shape = inputs.get_shape()

        if not shape.ndims == 3:
            raise Exception("Got ndims {} but expect a tensor of size "
                            " (?, n_steps, n_features)".format(shape.ndims))
        else:
            self.n_steps = shape[1]
            self.n_features = shape[2]

        # Unfold 3D matrix into a temporal sequence
        graph = tf.unstack(inputs, self.n_steps, 1)

        # Length of list outputs is n_steps
        # Size of outputs[0] is (?, n_hidden)
        outputs, states = tf.nn.static_rnn(self.cell,
                                           graph,
                                           dtype=tf.float32)


        ############################################################
        # Taken from
        #
        #   tensorlayer/blob/master/tensorlayer/layers.py
        #
        # TODO: If stacking multiple RNN layers, the output of the
        # first ones should be a cube matrix (?, n_steps, n_features)
        # and return_last = False (to implement in input arguments)
        #
        # Currently, only one RNN, so return output[-1]
        #
        ############################################################
        # if return_last:
        #     # 2D Tensor [batch_size, n_hidden]
        #     self.outputs = outputs[-1]
        # else:
        #     if return_seq_2d:
        #         # PTB tutorial: stack dense layer after that, or compute the cost from the output
        #         # 2D Tensor [n_example, n_hidden]
        #         try: # TF1.0
        #             self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden])
        #         except: # TF0.12
        #             self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden])


        #     else:
        #         # <akara>: stack more RNN layer after that
        #         # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
        #         try: # TF1.0
        #             self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_steps, n_hidden])
        #         except: # TF0.12
        #             self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, n_hidden])

        # Compare to tensorlayer, it is as if return_last = True
        return outputs[-1]
