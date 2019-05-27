#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from bob.learn.tensorflow.loss import triplet_loss
from . import check_features, get_trainable_variables

import logging

logger = logging.getLogger(__name__)


class Triplet(estimator.Estimator):
    """NN estimator for Triplet networks.

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A
    unified embedding for face recognition and clustering." Proceedings of the
    IEEE Conference on Computer Vision and Pattern Recognition. 2015.

    See :any:`Logits` for the description of parameters.
    """

    def __init__(
        self,
        architecture=None,
        optimizer=None,
        config=None,
        loss_op=triplet_loss,
        model_dir="",
        validation_batch_size=None,
        extra_checkpoint=None,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
    ):

        self.architecture = architecture
        self.optimizer = optimizer
        self.loss_op = loss_op
        self.loss = None
        self.extra_checkpoint = extra_checkpoint
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

        def _model_fn(features, labels, mode, params, config):

            if mode == tf.estimator.ModeKeys.TRAIN:

                # The input function needs to have dictionary pair with the `left` and `right` keys
                if (
                    "anchor" not in features.keys()
                    or "positive" not in features.keys()
                    or "negative" not in features.keys()
                ):
                    raise ValueError(
                        "The input function needs to contain a dictionary with the "
                        "keys `anchor`, `positive` and `negative` "
                    )

                # Building one graph
                trainable_variables = get_trainable_variables(self.extra_checkpoint)
                prelogits_anchor = self.architecture(
                    features["anchor"],
                    mode=mode,
                    trainable_variables=trainable_variables,
                )[0]
                prelogits_positive = self.architecture(
                    features["positive"],
                    reuse=True,
                    mode=mode,
                    trainable_variables=trainable_variables,
                )[0]
                prelogits_negative = self.architecture(
                    features["negative"],
                    reuse=True,
                    mode=mode,
                    trainable_variables=trainable_variables,
                )[0]

                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(
                        self.extra_checkpoint["checkpoint_path"],
                        self.extra_checkpoint["scopes"],
                    )

                # Compute Loss (for both TRAIN and EVAL modes)
                self.loss = self.loss_op(
                    prelogits_anchor, prelogits_positive, prelogits_negative
                )
                # Configure the Training Op (for TRAIN mode)
                global_step = tf.train.get_or_create_global_step()
                train_op = self.optimize_loss(
                    loss=self.loss,
                    global_step=global_step,
                    optimizer=self.optimizer,
                    learning_rate=self.optimize_loss_learning_rate,
                )
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, train_op=train_op
                )

            check_features(features)
            data = features["data"]

            # Compute the embeddings
            prelogits = self.architecture(data, mode=mode)[0]
            embeddings = tf.nn.l2_normalize(prelogits, 1)
            predictions = {"embeddings": embeddings}

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

        super(Triplet, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config
        )
