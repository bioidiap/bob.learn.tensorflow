#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 10 Aug 2016 16:38 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf

from .BaseLoss import BaseLoss
from bob.learn.tensorflow.utils import compute_euclidean_distance


class TripletFisherLoss(BaseLoss):
    """
    """

    def __init__(self, margin=0.2):
        self.margin = margin

    def __call__(self, anchor_embedding, positive_embedding, negative_embedding):

        with tf.name_scope("triplet_loss"):
            # Normalize
            anchor_embedding = tf.nn.l2_normalize(anchor_embedding, 1, 1e-10, name="anchor")
            positive_embedding = tf.nn.l2_normalize(positive_embedding, 1, 1e-10, name="positive")
            negative_embedding = tf.nn.l2_normalize(negative_embedding, 1, 1e-10, name="negative")

            average_class = tf.reduce_mean(anchor_embedding, 0)
            average_total = tf.div(tf.add(tf.reduce_mean(anchor_embedding, axis=0),\
                            tf.reduce_mean(negative_embedding, axis=0)), 2)

            length = anchor_embedding.get_shape().as_list()[0]
            split_positive = tf.unstack(positive_embedding, num=length, axis=0)
            split_negative = tf.unstack(negative_embedding, num=length, axis=0)

            Sw = None
            Sb = None
            for s in zip(split_positive, split_negative):
                positive = s[0]
                negative = s[1]

                buffer_sw = tf.reshape(tf.subtract(positive, average_class), shape=(2, 1))
                buffer_sw = tf.matmul(buffer_sw, tf.reshape(buffer_sw, shape=(1, 2)))

                buffer_sb = tf.reshape(tf.subtract(negative, average_total), shape=(2, 1))
                buffer_sb = tf.matmul(buffer_sb, tf.reshape(buffer_sb, shape=(1, 2)))

                if Sw is None:
                    Sw = buffer_sw
                    Sb = buffer_sb
                else:
                    Sw = tf.add(Sw, buffer_sw)
                    Sb = tf.add(Sb, buffer_sb)

            # Sw = tf.trace(Sw)
            # Sb = tf.trace(Sb)
            #loss = tf.trace(tf.div(Sb, Sw))
            loss = tf.trace(tf.div(Sw, Sb))

            return loss, tf.trace(Sb), tf.trace(Sw)
