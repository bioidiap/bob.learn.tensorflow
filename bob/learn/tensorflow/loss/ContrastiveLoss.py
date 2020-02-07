#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging
import tensorflow as tf
from bob.learn.tensorflow.utils import compute_euclidean_distance

logger = logging.getLogger(__name__)


def contrastive_loss(left_embedding, right_embedding, labels, contrastive_margin=2.0):
    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :math:`L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2`

    where, `0` are assign for pairs from the same class and `1` from pairs from different classes.


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
        right_embedding = tf.nn.l2_normalize(right_embedding, 1)

        d = compute_euclidean_distance(left_embedding, right_embedding)

        with tf.name_scope("within_class"):
            one = tf.constant(1.0)
            within_class = tf.multiply(one - labels, tf.square(d))  # (1-Y)*(d^2)
            within_class_loss = tf.reduce_mean(within_class, name="within_class")
            tf.add_to_collection(tf.GraphKeys.LOSSES, within_class_loss)

        with tf.name_scope("between_class"):
            max_part = tf.square(tf.maximum(contrastive_margin - d, 0))
            between_class = tf.multiply(
                labels, max_part
            )  # (Y) * max((margin - d)^2, 0)
            between_class_loss = tf.reduce_mean(between_class, name="between_class")
            tf.add_to_collection(tf.GraphKeys.LOSSES, between_class_loss)

        with tf.name_scope("total_loss"):
            loss = 0.5 * (within_class + between_class)
            loss = tf.reduce_mean(loss, name="contrastive_loss")
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

        tf.summary.scalar("contrastive_loss", loss)
        tf.summary.scalar("between_class", between_class_loss)
        tf.summary.scalar("within_class", within_class_loss)

        return loss
