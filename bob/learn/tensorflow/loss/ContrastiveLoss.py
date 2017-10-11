#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 10 Aug 2016 16:38 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf

from .BaseLoss import BaseLoss
from bob.learn.tensorflow.utils import compute_euclidean_distance


class ContrastiveLoss(BaseLoss):
    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :math:`L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2`

    **Parameters**

    left_feature:
      First element of the pair

    right_feature:
      Second element of the pair

    label:
      Label of the pair (0 or 1)

    margin:
      Contrastive margin

    """

    def __init__(self, contrastive_margin=1.0):
        self.contrastive_margin = contrastive_margin

    def __call__(self, label, left_feature, right_feature):
        with tf.name_scope("contrastive_loss"):
            label = tf.to_float(label)
            
            left_feature = tf.nn.l2_normalize(left_feature, 1)
            right_feature  = tf.nn.l2_normalize(right_feature, 1)

            one = tf.constant(1.0)

            d = compute_euclidean_distance(left_feature, right_feature)
            within_class = tf.multiply(one - label, tf.square(d))  # (1-Y)*(d^2)
            
            
            max_part = tf.square(tf.maximum(self.contrastive_margin - d, 0))
            between_class = tf.multiply(label, max_part)  # (Y) * max((margin - d)^2, 0)

            loss = 0.5 * (within_class + between_class)

            loss_dict = dict()
            loss_dict['loss'] = tf.reduce_mean(loss)
            loss_dict['between_class'] = tf.reduce_mean(between_class)
            loss_dict['within_class'] = tf.reduce_mean(within_class)

            return loss_dict
