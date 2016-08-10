#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 10 Aug 2016 16:38 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf
from bob.learn.tensorflow.loss.Baseloss as Baseloss


class ContrastiveLoss(BaseLoss):
    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2

    OR MAYBE THAT

    L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    """

    def __init__(self, label, left_feature, right_feature, margin):

        with tf.name_scope("contrastive_loss"):
            label = tf.to_float(label)
            one = tf.constant(1.0)

            d = bob.learn.tensorflow.util.compute_euclidean_distance(left_feature, right_feature)
            # first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
            # first_part = tf.mul(label, tf.square(d))  # (Y-1)*(d^2)
            between_class = tf.exp(tf.mul(one - label, tf.square(d)))  # (1-Y)*(d^2)
            max_part = tf.square(tf.maximum(margin - d, 0))

            within_class = tf.mul(label, max_part)  # (Y) * max((margin - d)^2, 0)
            # second_part = tf.mul(one-label, max_part)  # (Y) * max((margin - d)^2, 0)

            self.loss = 0.5 * tf.reduce_mean(within_class + between_class)

            self.within_class = tf.reduce_mean(within_class), tf.reduce_mean(between_class)
