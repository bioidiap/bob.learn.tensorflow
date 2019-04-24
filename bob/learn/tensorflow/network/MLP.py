#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.learn.tensorflow.network.utils import is_trainable

slim = tf.contrib.slim


def mlp(
    inputs,
    output_shape,
    hidden_layers=[10],
    hidden_activation=tf.nn.tanh,
    output_activation=None,
    seed=10,
    **kwargs
):
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

    initializer = tf.contrib.layers.xavier_initializer(
        uniform=False, dtype=tf.float32, seed=seed
    )

    graph = inputs
    for i in range(len(hidden_layers)):

        weights = hidden_layers[i]
        graph = slim.fully_connected(
            graph,
            weights,
            weights_initializer=initializer,
            activation_fn=hidden_activation,
            scope="fc_{0}".format(i),
        )

    graph = slim.fully_connected(
        graph,
        output_shape,
        weights_initializer=initializer,
        activation_fn=output_activation,
        scope="fc_output",
    )

    return graph


def mlp_with_batchnorm_and_dropout(
    inputs,
    fully_connected_layers,
    mode=tf.estimator.ModeKeys.TRAIN,
    trainable_variables=None,
    **kwargs
):

    if trainable_variables is not None:
        raise ValueError(
            "The batch_norm layers selectable training is not implemented!"
        )

    end_points = {}
    net = slim.flatten(inputs)

    weight_decay = 1e-5
    dropout_keep_prob = 0.5
    batch_norm_params = {
        # Decay for the moving averages.
        "decay": 0.995,
        # epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # force in-place updates of mean and variance estimates
        "updates_collections": None,
        "is_training": (mode == tf.estimator.ModeKeys.TRAIN),
    }

    with slim.arg_scope(
        [slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
    ), tf.name_scope("MLP"):

        # hidden layers
        for i, n in enumerate(fully_connected_layers):
            name = "fc_{:0d}".format(i)
            trainable = is_trainable(name, trainable_variables, mode=mode)
            with slim.arg_scope(
                [slim.batch_norm], is_training=trainable, trainable=trainable
            ):

                net = slim.fully_connected(net, n, scope=name, trainable=trainable)
                end_points[name] = net

            name = "dropout_{:0d}".format(i)
            net = slim.dropout(
                net,
                dropout_keep_prob,
                is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                scope=name,
            )
            end_points[name] = net

    return net, end_points
