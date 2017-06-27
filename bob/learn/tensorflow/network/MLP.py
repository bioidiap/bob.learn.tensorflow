#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the lenet architecture
"""

import tensorflow as tf


class MLP(object):
    """An MLP is a representation of a Multi-Layer Perceptron.

    This implementation is feed-forward and fully-connected.
    The implementation allows setting a global and the output activation functions.
    References to fully-connected feed-forward networks: Bishop's Pattern Recognition and Machine Learning, Chapter 5. Figure 5.1 shows what is programmed.

    MLPs normally are multi-layered systems, with 1 or more hidden layers.

    **Parameters**

        output_shape: number of neurons in the output.

        hidden_layers: :py:class:`list` that contains the amount of hidden layers, where each element is the number of neurons

        hidden_activation: Activation function of the hidden layers. Possible values can be seen
                          `here <https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#activation-functions>`_.
                           If you set to ``None``, the activation will be linear.

        output_activation: Activation of the output layer.  If you set to `None`, the activation will be linear

        seed: 

        device:
    """
    def __init__(self,
                 output_shape,
                 hidden_layers=[10],
                 hidden_activation=tf.nn.tanh,
                 output_activation=None,
                 seed=10,
                 device="/cpu:0"):

        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.seed = seed
        self.device = device

    def __call__(self, inputs):
        slim = tf.contrib.slim
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=self.seed)

        #if (not (isinstance(hidden_layers, list) or isinstance(hidden_layers, tuple))) or len(hidden_layers) == 0:
        #    raise ValueError("Invalid input for hidden_layers: {0} ".format(hidden_layers))

        graph = inputs
        for i in range(len(self.hidden_layers)):

            weights = self.hidden_layers[i]
            graph = slim.fully_connected(graph, weights,
                                         weights_initializer=initializer,
                                         activation_fn=self.hidden_activation,
                                         scope='fc_{0}'.format(i))

        graph = slim.fully_connected(graph, self.output_shape,
                                     weights_initializer=initializer,
                                     activation_fn=self.output_activation,
                                     scope='fc_output')

        return graph
