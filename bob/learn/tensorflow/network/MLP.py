#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf


def mlp(inputs,
        output_shape,
        hidden_layers=[10],
        hidden_activation=tf.nn.tanh,
        output_activation=None,
        seed=10,
        **kwargs):
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
    """

    slim = tf.contrib.slim
    initializer = tf.contrib.layers.xavier_initializer(
        uniform=False, dtype=tf.float32, seed=seed)

    graph = inputs
    for i in range(len(hidden_layers)):

        weights = hidden_layers[i]
        graph = slim.fully_connected(
            graph,
            weights,
            weights_initializer=initializer,
            activation_fn=hidden_activation,
            scope='fc_{0}'.format(i))

    graph = slim.fully_connected(
        graph,
        output_shape,
        weights_initializer=initializer,
        activation_fn=output_activation,
        scope='fc_output')

    return graph
