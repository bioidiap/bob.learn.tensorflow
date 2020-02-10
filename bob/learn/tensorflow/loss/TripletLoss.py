#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging
logger = logging.getLogger(__name__)
import tensorflow as tf

from bob.learn.tensorflow.utils import compute_euclidean_distance


def triplet_loss(anchor_embedding,
                 positive_embedding,
                 negative_embedding,
                 margin=5.0):
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

    with tf.name_scope("triplet_loss"):
        # Normalize
        anchor_embedding = tf.nn.l2_normalize(
            anchor_embedding, 1, 1e-10, name="anchor")
        positive_embedding = tf.nn.l2_normalize(
            positive_embedding, 1, 1e-10, name="positive")
        negative_embedding = tf.nn.l2_normalize(
            negative_embedding, 1, 1e-10, name="negative")

        d_positive = tf.reduce_sum(
            tf.square(tf.subtract(anchor_embedding, positive_embedding)), 1)
        d_negative = tf.reduce_sum(
            tf.square(tf.subtract(anchor_embedding, negative_embedding)), 1)

        basic_loss = tf.add(tf.subtract(d_positive, d_negative), margin)

        with tf.name_scope("TripletLoss"):
            # Between
            between_class_loss = tf.reduce_mean(d_negative)
            tf.summary.scalar('loss_between_class', between_class_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, between_class_loss)

            # Within
            within_class_loss = tf.reduce_mean(d_positive)
            tf.summary.scalar('loss_within_class', within_class_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, within_class_loss)

            # Total loss
            loss = tf.reduce_mean(
                tf.maximum(basic_loss, 0.0), 0, name="total_loss")
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
            tf.summary.scalar('loss_triplet', loss)

        return loss


def triplet_fisher_loss(anchor_embedding, positive_embedding,
                        negative_embedding):

    with tf.name_scope("triplet_loss"):
        # Normalize
        anchor_embedding = tf.nn.l2_normalize(
            anchor_embedding, 1, 1e-10, name="anchor")
        positive_embedding = tf.nn.l2_normalize(
            positive_embedding, 1, 1e-10, name="positive")
        negative_embedding = tf.nn.l2_normalize(
            negative_embedding, 1, 1e-10, name="negative")

        average_class = tf.reduce_mean(anchor_embedding, 0)
        average_total = tf.div(tf.add(tf.reduce_mean(anchor_embedding, axis=0),\
                        tf.reduce_mean(negative_embedding, axis=0)), 2)

        length = anchor_embedding.get_shape().as_list()[0]
        dim = anchor_embedding.get_shape().as_list()[1]
        split_positive = tf.unstack(positive_embedding, num=length, axis=0)
        split_negative = tf.unstack(negative_embedding, num=length, axis=0)

        Sw = None
        Sb = None
        for s in zip(split_positive, split_negative):
            positive = s[0]
            negative = s[1]

            buffer_sw = tf.reshape(
                tf.subtract(positive, average_class), shape=(dim, 1))
            buffer_sw = tf.matmul(buffer_sw,
                                  tf.reshape(buffer_sw, shape=(1, dim)))

            buffer_sb = tf.reshape(
                tf.subtract(negative, average_total), shape=(dim, 1))
            buffer_sb = tf.matmul(buffer_sb,
                                  tf.reshape(buffer_sb, shape=(1, dim)))

            if Sw is None:
                Sw = buffer_sw
                Sb = buffer_sb
            else:
                Sw = tf.add(Sw, buffer_sw)
                Sb = tf.add(Sb, buffer_sb)

        # Sw = tf.trace(Sw)
        # Sb = tf.trace(Sb)
        #loss = tf.trace(tf.div(Sb, Sw))
        loss = tf.trace(tf.div(Sw, Sb), name=tf.GraphKeys.LOSSES)

        return loss, tf.trace(Sb), tf.trace(Sw)


def triplet_average_loss(anchor_embedding,
                         positive_embedding,
                         negative_embedding,
                         margin=5.0):
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

    with tf.name_scope("triplet_loss"):
        # Normalize
        anchor_embedding = tf.nn.l2_normalize(
            anchor_embedding, 1, 1e-10, name="anchor")
        positive_embedding = tf.nn.l2_normalize(
            positive_embedding, 1, 1e-10, name="positive")
        negative_embedding = tf.nn.l2_normalize(
            negative_embedding, 1, 1e-10, name="negative")

        anchor_mean = tf.reduce_mean(anchor_embedding, 0)

        d_positive = tf.reduce_sum(
            tf.square(tf.subtract(anchor_mean, positive_embedding)), 1)
        d_negative = tf.reduce_sum(
            tf.square(tf.subtract(anchor_mean, negative_embedding)), 1)

        basic_loss = tf.add(tf.subtract(d_positive, d_negative), margin)
        loss = tf.reduce_mean(
            tf.maximum(basic_loss, 0.0), 0, name=tf.GraphKeys.LOSSES)

        return loss, tf.reduce_mean(d_negative), tf.reduce_mean(d_positive)


