#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf

from bob.learn.tensorflow.utils import compute_euclidean_distance


def contrastive_loss(left_embedding, right_embedding, labels, contrastive_margin=1.0):
    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :math:`L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2`

    **Parameters**

    left_feature:
      First element of the pair

    right_feature:
      Second element of the pair

    labels:
      Label of the pair (0 or 1)

    margin:
      Contrastive margin

    """

    with tf.name_scope("contrastive_loss"):
        labels = tf.to_float(labels)
        
        left_embedding = tf.nn.l2_normalize(left_embedding, 1)
        right_embedding  = tf.nn.l2_normalize(right_embedding, 1)

        one = tf.constant(1.0)

        d = compute_euclidean_distance(left_embedding, right_embedding)
        within_class = tf.multiply(one - labels, tf.square(d))  # (1-Y)*(d^2)
        
        max_part = tf.square(tf.maximum(contrastive_margin - d, 0))
        between_class = tf.multiply(labels, max_part)  # (Y) * max((margin - d)^2, 0)

        loss =  0.5 * (within_class + between_class)

        loss_dict = dict()
        loss_dict['loss'] = tf.reduce_mean(loss, name=tf.GraphKeys.LOSSES)
        loss_dict['between_class'] = tf.reduce_mean(between_class, name=tf.GraphKeys.LOSSES)
        loss_dict['within_class'] = tf.reduce_mean(within_class, name=tf.GraphKeys.LOSSES)

        return loss_dict
