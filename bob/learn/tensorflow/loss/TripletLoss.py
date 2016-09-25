#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 10 Aug 2016 16:38 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf

from .BaseLoss import BaseLoss
from bob.learn.tensorflow.util import compute_euclidean_distance


class TripletLoss(BaseLoss):
    """
    Compute the triplet loss as in

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    "Facenet: A unified embedding for face recognition and clustering."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

    L  = sum(  |f_a - f_p|^2 - |f_a - f_n|^2  + \lambda)


    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    """

    def __init__(self, margin=2.0):
        self.margin = margin

    def __call__(self, anchor_feature, positive_feature, negative_feature):

        with tf.name_scope("triplet_loss"):

            d_positive = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
            d_negative = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

            loss = tf.maximum(0., d_positive - d_negative + self.margin)

            return tf.reduce_mean(loss), tf.reduce_mean(d_positive), tf.reduce_mean(d_negative)
