#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


# def mean_cross_entropy_loss(logits, labels, add_regularization_losses=True):
#     """
#     Simple CrossEntropy loss.
#     Basically it wrapps the function tf.nn.sparse_softmax_cross_entropy_with_logits.

#     **Parameters**
#       logits:
#       labels:
#       add_regularization_losses: Regulize the loss???

#     """

#     with tf.compat.v1.variable_scope('cross_entropy_loss'):
#         cross_loss = tf.reduce_mean(
#             input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
#                 logits=logits, labels=labels),
#             name="cross_entropy_loss")

#         tf.compat.v1.summary.scalar('cross_entropy_loss', cross_loss)
#         tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.LOSSES, cross_loss)

#         if add_regularization_losses:
#             regularization_losses = tf.compat.v1.get_collection(
#                 tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

#             total_loss = tf.add_n(
#                 [cross_loss] + regularization_losses, name="total_loss")
#             return total_loss
#         else:
#             return cross_loss


def mean_cross_entropy_center_loss(
    logits, prelogits, labels, n_classes, alpha=0.9, factor=0.01
):
    """
    Implementation of the CrossEntropy + Center Loss from the paper
    "A Discriminative Feature Learning Approach for Deep Face Recognition"(http://ydwen.github.io/papers/WenECCV16.pdf)

    **Parameters**
      logits:
      prelogits:
      labels:
      n_classes: Number of classes of your task
      alpha: Alpha factor ((1-alpha)*centers-prelogits)
      factor: Weight factor of the center loss

    """
    # Cross entropy
    with tf.compat.v1.variable_scope("cross_entropy_loss"):
        cross_loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            ),
            name="cross_entropy_loss",
        )
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.LOSSES, cross_loss)
        tf.compat.v1.summary.scalar("loss_cross_entropy", cross_loss)

    # Appending center loss
    with tf.compat.v1.variable_scope("center_loss"):
        n_features = prelogits.get_shape()[1]

        centers = tf.compat.v1.get_variable(
            "centers",
            [n_classes, n_features],
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(0),
            trainable=False,
        )

        # label = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        diff = (1 - alpha) * (centers_batch - prelogits)
        centers = tf.compat.v1.scatter_sub(centers, labels, diff)
        center_loss = tf.reduce_mean(input_tensor=tf.square(prelogits - centers_batch))
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, center_loss * factor
        )
        tf.compat.v1.summary.scalar("loss_center", center_loss)

    # Adding the regularizers in the loss
    with tf.compat.v1.variable_scope("total_loss"):
        regularization_losses = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES
        )
        total_loss = tf.add_n([cross_loss] + regularization_losses, name="total_loss")
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.LOSSES, total_loss)
        tf.compat.v1.summary.scalar("loss_total", total_loss)

    loss = dict()
    loss["loss"] = total_loss
    loss["centers"] = centers

    return loss
