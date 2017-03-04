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

    def __init__(self, margin=0.2):
        self.margin = margin

    def body(self, mean, x):
        buffer = mean - x
        return tf.matmul(buffer, tf.transpose(buffer))

    """
    def cond(i):
        return tf.reduce_sum(i) < 10

    def body(i):
        return tf.add(i, 1)

    i = tf.placeholder(tf.float32)
    op = tf.while_loop(cond, body, [i])
    print(session.run(op, feed_dict={i: 0}))
    """

    def __call__(self, anchor_embedding, positive_embedding, negative_embedding):

        with tf.name_scope("triplet_loss"):
            # Normalize
            anchor_embedding = tf.nn.l2_normalize(anchor_embedding, 1, 1e-10, name="anchor")
            positive_embedding = tf.nn.l2_normalize(positive_embedding, 1, 1e-10, name="positive")
            negative_embedding = tf.nn.l2_normalize(negative_embedding, 1, 1e-10, name="negative")

            #anchor_mean = tf.reduce_mean(anchor_embedding, 0)
            #result = tf.while_loop(condition, self.body(anchor_mean), [positive_embedding])


            #p_minus_mean = tf.subtract(anchor_mean, positive_embedding)
            #s_w = tf.divide(tf.matmul(tf.transpose(p_minus_mean), p_minus_mean), 1)

            #s_w = tf.trace(tf.reduce_mean(tf.square(tf.subtract(anchor_mean, positive_embedding)), 1))
            #s_b = tf.trace(tf.reduce_mean(tf.square(tf.subtract(anchor_mean, negative_embedding)), 1))

            #s_w = tf.reduce_mean(tf.square(tf.subtract(anchor_mean, positive_embedding)), 1)

            #loss = s_w/s_b

            #return s_w, p_minus_mean
            #return tf.multiply(p_minus_mean, tf.transpose(p_minus_mean))
