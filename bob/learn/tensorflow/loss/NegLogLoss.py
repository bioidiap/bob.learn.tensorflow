#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 19 Oct 23:43:22 2016

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf

from .BaseLoss import BaseLoss


class NegLogLoss(BaseLoss):
    """
    Compute the negative log likelihood loss
    This is similar to the combination of LogSoftMax layer and ClassNLLCriterion in Torch7
    """

    def __init__(self, operation):
        # loss function is None since we compute the custom one inside __call__()
        super(NegLogLoss, self).__init__(None, operation)

    def gather_nd(self, params, indices, name=None):
        shape = params.get_shape().as_list()
        rank = len(shape)
        flat_params = tf.reshape(params, [-1])
        if rank > 2:
            indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + range(0, rank - 1), name))
        elif rank == 2:
            indices_unpacked = tf.unstack(indices)
        else:
            indices_unpacked = indices
        flat_indices = [i * rank + indices_unpacked[i] for i in range(0, len(indices_unpacked))]
        return tf.gather(flat_params, flat_indices, name=name)

    def __call__(self, graph, label):
        # get the log-probabilities with log softmax
        log_probabilities = tf.nn.log_softmax(graph)
        # negative of the log-probability that correspond to the correct label
        correct_probabilities = self.gather_nd(log_probabilities, label)
        neg_log_prob = tf.negative(correct_probabilities)
        # use negative log likelihood as the loss
        return self.operation(neg_log_prob)
