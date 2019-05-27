#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from . import check_features, get_trainable_variables

import logging

logger = logging.getLogger(__name__)


class Siamese(estimator.Estimator):
    """NN estimator for Siamese Networks.
    Proposed in: "Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a
    similarity metric discriminatively, with application to face verification."
    Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer
    Society Conference on. Vol. 1. IEEE, 2005."

    See :any:`Logits` for the description of parameters.
    """

    def __init__(
        self,
        architecture=None,
        optimizer=None,
        config=None,
        loss_op=None,
        model_dir="",
        validation_batch_size=None,
        params=None,
        extra_checkpoint=None,
        add_histograms=None,
        add_regularization_losses=True,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
    ):

        self.architecture = architecture
        self.optimizer = optimizer
        self.loss_op = loss_op
        self.loss = None
        self.extra_checkpoint = extra_checkpoint
        self.add_regularization_losses = add_regularization_losses

        self.optimize_loss = optimize_loss
        self.optimize_loss_learning_rate = optimize_loss_learning_rate

        if self.architecture is None:
            raise ValueError("Please specify a function to build the architecture !!")

        if self.optimizer is None:
            raise ValueError(
                "Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!"
            )

        if self.loss_op is None:
            raise ValueError("Please specify a function to build the loss !!")

        def _model_fn(features, labels, mode):
            if mode == tf.estimator.ModeKeys.TRAIN:

                # Building one graph, by default everything is trainable
                # The input function needs to have dictionary pair with the `left` and `right` keys
                if "left" not in features.keys() or "right" not in features.keys():
                    raise ValueError(
                        "The input function needs to contain a dictionary with the keys `left` and `right` "
                    )

                # Building one graph
                trainable_variables = get_trainable_variables(self.extra_checkpoint)
                data_left = features["left"]
                data_left = (
                    data_left["data"] if isinstance(data_left, dict) else data_left
                )
                data_right = features["right"]
                data_right = (
                    data_right["data"] if isinstance(data_right, dict) else data_right
                )
                prelogits_left, end_points_left = self.architecture(
                    data_left, mode=mode, trainable_variables=trainable_variables
                )
                prelogits_right, end_points_right = self.architecture(
                    data_right,
                    reuse=True,
                    mode=mode,
                    trainable_variables=trainable_variables,
                )

                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(
                        self.extra_checkpoint["checkpoint_path"],
                        self.extra_checkpoint["scopes"],
                    )

                # Compute Loss (for both TRAIN and EVAL modes)
                labels = (
                    tf.not_equal(labels["left"], labels["right"])
                    if isinstance(labels, dict)
                    else labels
                )
                self.loss = self.loss_op(prelogits_left, prelogits_right, labels)
                if self.add_regularization_losses:
                    regularization_losses = tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES
                    )
                    regularization_losses = [
                        tf.cast(l, self.loss.dtype) for l in regularization_losses
                    ]
                    self.loss = tf.add_n(
                        [self.loss] + regularization_losses, name="total_loss"
                    )
                train_op = self.optimize_loss(
                    loss=self.loss,
                    global_step=tf.train.get_or_create_global_step(),
                    optimizer=self.optimizer,
                    learning_rate=self.optimize_loss_learning_rate,
                )

                # add histograms summaries
                if add_histograms == "all":
                    for v in tf.all_variables():
                        tf.summary.histogram(v.name, v)
                elif add_histograms == "train":
                    for v in tf.trainable_variables():
                        tf.summary.histogram(v.name, v)

                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, train_op=train_op
                )

            check_features(features)
            data = features["data"]
            key = features["key"]

            # Compute the embeddings
            prelogits = self.architecture(data, mode=mode)[0]
            embeddings = tf.nn.l2_normalize(prelogits, 1)
            predictions = {"embeddings": embeddings, "key": key}

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            predictions_op = predict_using_tensors(
                predictions["embeddings"], labels, num=validation_batch_size
            )
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions_op
                )
            }

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=tf.reduce_mean(1), eval_metric_ops=eval_metric_ops
            )

        super(Siamese, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, params=params, config=config
        )
