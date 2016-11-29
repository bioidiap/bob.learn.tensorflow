#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 10 Aug 2016 16:38 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf

from .BaseLoss import BaseLoss
from bob.learn.tensorflow.utils import compute_euclidean_distance


class TripletLoss(BaseLoss):
    """
    Compute the triplet loss as in

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    "Facenet: A unified embedding for face recognition and clustering."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

    :math:`L  = sum(  |f_a - f_p|^2 - |f_a - f_n|^2  + \lambda)`

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

    def __init__(self, margin=5.0):
        self.margin = margin

    def __call__(self, anchor_embedding, positive_embedding, negative_embedding):

        with tf.name_scope("triplet_loss"):
            # Normalize
            anchor_embedding = tf.nn.l2_normalize(anchor_embedding, 1, 1e-10)
            positive_embedding = tf.nn.l2_normalize(positive_embedding, 1, 1e-10)
            negative_embedding = tf.nn.l2_normalize(negative_embedding, 1, 1e-10)

            d_positive = tf.reduce_sum(tf.square(tf.sub(anchor_embedding, positive_embedding)), 1)
            d_negative = tf.reduce_sum(tf.square(tf.sub(anchor_embedding, negative_embedding)), 1)

            basic_loss = tf.add(tf.sub(d_positive, d_negative), self.margin)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

            return loss, tf.reduce_mean(d_negative), tf.reduce_mean(d_positive)
