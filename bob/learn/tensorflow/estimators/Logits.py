#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from bob.learn.tensorflow.network.utils import append_logits
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from bob.learn.tensorflow.loss import (
    mean_cross_entropy_center_loss,
    balanced_softmax_cross_entropy_loss_weights,
    balanced_sigmoid_cross_entropy_loss_weights,
)

from . import check_features, get_trainable_variables

import logging

logger = logging.getLogger(__name__)


class Logits(estimator.Estimator):
    """Logits estimator.

    NN estimator with `Cross entropy loss
    <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_
    in the hot-encoded layer
    :py:class:`bob.learn.tensorflow.estimators.Logits`.

    The architecture function should follow the following pattern::

      def my_beautiful_architecture(placeholder, **kwargs):

        end_points = dict()
        graph = convXX(placeholder)
        end_points['conv'] = graph

      return graph, end_points


    The **loss** function should follow the following pattern::

      def my_beautiful_loss(logits, labels, **kwargs):
        return loss_set_of_ops(logits, labels)


    Attributes
    ----------

      architecture:
         Pointer to a function that builds the graph.

      optimizer:
         One of the tensorflow solvers

      config:

      n_classes:
         Number of classes of your problem. The logits will be appended in this
         class

      loss_op:
         Pointer to a function that computes the loss.

      embedding_validation:
         Run the validation using embeddings?? [default: False]

      model_dir:
        Model path

      validation_batch_size:
        Size of the batch for validation. This value is used when the
        validation with embeddings is used. This is a hack.

      params:
        Extra params for the model function (please see
        https://www.tensorflow.org/extend/estimators for more info)

      extra_checkpoint: dict
        In case you want to use other model to initialize some variables.
        This argument should be in the following format::

          extra_checkpoint = {
            "checkpoint_path": <YOUR_CHECKPOINT>,
            "scopes": dict({"<SOURCE_SCOPE>/": "<TARGET_SCOPE>/"}),
            "trainable_variables": [<LIST OF VARIABLES OR SCOPES THAT YOU WANT TO RETRAIN>]
          }

      apply_moving_averages: bool
        Apply exponential moving average in the training variables and in the loss.
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        By default the decay for the variable averages is 0.9999 and for the loss is 0.9
    """

    def __init__(
        self,
        architecture,
        optimizer,
        loss_op,
        n_classes,
        config=None,
        embedding_validation=False,
        model_dir="",
        validation_batch_size=None,
        params=None,
        extra_checkpoint=None,
        apply_moving_averages=True,
        add_histograms=None,
        vat_loss=None,
        architecture_has_logits=False,
        balanced_loss_weight=False,
        use_sigmoid=False,
        labels_are_one_hot=False,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
    ):

        self.architecture = architecture
        self.n_classes = n_classes
        self.loss_op = loss_op
        self.loss = None
        self.embedding_validation = embedding_validation
        self.extra_checkpoint = extra_checkpoint
        self.apply_moving_averages = apply_moving_averages

        if self.apply_moving_averages and isinstance(optimizer, tf.train.Optimizer):
            logger.info("Encapsulating the optimizer with the MovingAverageOptimizer")
            optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

        self.optimizer = optimizer
        self.vat_loss = vat_loss
        self.balanced_loss_weight = balanced_loss_weight
        self.use_sigmoid = use_sigmoid
        self.labels_are_one_hot = labels_are_one_hot
        self.optimize_loss = optimize_loss
        self.optimize_loss_learning_rate = optimize_loss_learning_rate

        if self.n_classes < 3 and not self.use_sigmoid:
            logger.error(
                "n_classes is less than 3 and use_sigmoid is False. Are you sure? "
                "It's better to use sigmoid activation on your logits for binary "
                "classification tasks. Don't forget to change your loss to a sigmoid "
                "loss too! Change n_classes to 1 too if not already."
            )

        def _model_fn(features, labels, mode, config):

            check_features(features)
            data = features["data"]
            key = features["key"]

            # Checking if we have some variables/scope that we may want to shut
            # down
            trainable_variables = get_trainable_variables(
                self.extra_checkpoint, mode=mode
            )
            prelogits, end_points = self.architecture(
                data, mode=mode, trainable_variables=trainable_variables
            )
            if architecture_has_logits:
                logits, prelogits = prelogits, end_points["prelogits"]
            else:
                logits = append_logits(
                    prelogits, n_classes, trainable_variables=trainable_variables
                )

            # for vat_loss
            def whole_architecture(features, mode, reuse):
                prelogits, end_points = self.architecture(
                    features,
                    mode=mode,
                    trainable_variables=trainable_variables,
                    reuse=reuse,
                )
                logits = append_logits(
                    prelogits,
                    n_classes,
                    reuse=reuse,
                    trainable_variables=trainable_variables,
                )
                return logits, end_points

            if architecture_has_logits:
                whole_architecture = self.architecture

            if self.embedding_validation and mode != tf.estimator.ModeKeys.TRAIN:

                # Compute the embeddings
                embeddings = tf.nn.l2_normalize(prelogits, 1)
                predictions = {"embeddings": embeddings, "key": key}
            else:
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1)
                    if self.n_classes > 1
                    else logits >= 0.5,
                    # Add `softmax_tensor` to the graph. It is used for PREDICT
                    # and by the `logging_hook`.
                    "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
                    if self.use_sigmoid
                    else tf.nn.softmax(logits, name="softmax_tensor"),
                    "key": key,
                }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            if self.embedding_validation and mode != tf.estimator.ModeKeys.TRAIN:
                predictions_op = predict_using_tensors(
                    predictions["embeddings"], labels, num=validation_batch_size
                )
            else:
                predictions_op = predictions["classes"]

            accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions_op)
            metrics = {"accuracy": accuracy}

            loss_kwargs = {}
            loss_kwargs["logits"] = logits
            if self.balanced_loss_weight:
                weights_fn = (
                    balanced_sigmoid_cross_entropy_loss_weights
                    if self.use_sigmoid
                    else balanced_softmax_cross_entropy_loss_weights
                )
                labels_one_hot = labels
                if (not self.labels_are_one_hot) and (not self.use_sigmoid):
                    labels_one_hot = tf.one_hot(labels, depth=self.n_classes)
                weights = weights_fn(labels_one_hot, logits.dtype)
                loss_kwargs["weights"] = weights

            if self.use_sigmoid:
                loss_kwargs["multi_class_labels"] = labels
            else:
                loss_kwargs["labels"] = labels

            if mode == tf.estimator.ModeKeys.EVAL:
                self.loss = self.loss_op(**loss_kwargs)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=self.loss,
                    train_op=None,
                    eval_metric_ops=metrics,
                )

            # restore the model from an extra_checkpoint
            if extra_checkpoint is not None:
                if (
                    "Logits/" not in extra_checkpoint["scopes"]
                    and not architecture_has_logits
                ):
                    logger.warning(
                        '"Logits/" (which are automatically added by this '
                        "Logits class are not in the scopes of "
                        "extra_checkpoint). Did you mean to restore the "
                        "Logits variables as well?"
                    )
                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=extra_checkpoint["checkpoint_path"],
                    assignment_map=extra_checkpoint["scopes"],
                )

            # Calculate Loss
            self.loss = self.loss_op(**loss_kwargs)

            if self.vat_loss is not None:
                vat_loss = self.vat_loss(data, logits, whole_architecture, mode)
                self.loss = tf.add_n([self.loss, vat_loss], name="total_loss_with_vat")

            # Compute the moving average of all individual losses and the total loss.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
            loss_averages_op = loss_averages.apply(
                tf.get_collection(tf.GraphKeys.LOSSES)
            )

            train_op = tf.group(
                self.optimize_loss(
                    loss=self.loss,
                    global_step=tf.train.get_or_create_global_step(),
                    optimizer=self.optimizer,
                    learning_rate=self.optimize_loss_learning_rate,
                ),
                loss_averages_op,
            )

            # Get the moving average saver after optimizer.minimize is called
            if self.apply_moving_averages:
                self.saver, self.scaffold = moving_average_scaffold(
                    self.optimizer.optimizer
                    if hasattr(self.optimizer, "optimizer")
                    else self.optimizer,
                    config,
                )
            else:
                self.saver, self.scaffold = None, None

            # Log accuracy and loss
            with tf.name_scope("train_metrics"):
                tf.summary.scalar("accuracy", accuracy[1])
                for l in tf.get_collection(tf.GraphKeys.LOSSES):
                    tf.summary.scalar(l.op.name + "_averaged", loss_averages.average(l))

            # add histograms summaries
            if add_histograms == "all":
                for v in tf.all_variables():
                    tf.summary.histogram(v.name, v)
            elif add_histograms == "train":
                for v in tf.trainable_variables():
                    tf.summary.histogram(v.name, v)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=self.loss,
                train_op=train_op,
                eval_metric_ops=metrics,
                scaffold=self.scaffold,
            )

        super(Logits, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, params=params, config=config
        )


class LogitsCenterLoss(estimator.Estimator):
    """Logits estimator with center loss.

    NN estimator with `Cross entropy loss
    <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_
    in the hot-encoded layer :py:class:`bob.learn.tensorflow.estimators.Logits`
    plus the center loss implemented in: "Wen, Yandong, et al. "A
    discriminative feature learning approach for deep face recognition."
    European Conference on Computer Vision. Springer, Cham, 2016."

    See :any:`Logits` for the description of parameters.
    """

    def __init__(
        self,
        architecture=None,
        optimizer=None,
        config=None,
        n_classes=0,
        embedding_validation=False,
        model_dir="",
        alpha=0.9,
        factor=0.01,
        validation_batch_size=None,
        params=None,
        extra_checkpoint=None,
        apply_moving_averages=True,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
    ):

        self.architecture = architecture
        self.n_classes = n_classes
        self.alpha = alpha
        self.factor = factor
        self.loss = None
        self.embedding_validation = embedding_validation
        self.extra_checkpoint = extra_checkpoint
        self.optimize_loss = optimize_loss
        self.optimize_loss_learning_rate = optimize_loss_learning_rate
        self.apply_moving_averages = apply_moving_averages

        if self.architecture is None:
            raise ValueError("Please specify a function to build the architecture !!")

        if optimizer is None:
            raise ValueError(
                "Please specify a optimizer (https://www.tensorflow.org/"
                "api_guides/python/train) !!"
            )

        if self.n_classes <= 0:
            raise ValueError("Number of classes must be greated than 0")

        if self.apply_moving_averages and isinstance(optimizer, tf.train.Optimizer):
            logger.info("Encapsulating the optimizer with the MovingAverageOptimizer")
            optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)
        self.optimizer = optimizer

        def _model_fn(features, labels, mode, config):

            check_features(features)
            data = features["data"]
            key = features["key"]

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Building the training graph

                # Checking if we have some variables/scope that we may want to shut down
                trainable_variables = get_trainable_variables(self.extra_checkpoint)
                prelogits = self.architecture(
                    data, mode=mode, trainable_variables=trainable_variables
                )[0]
                logits = append_logits(prelogits, n_classes)

                global_step = tf.train.get_or_create_global_step()

                # Compute the moving average of all individual losses and the total loss.
                if self.apply_moving_averages:
                    variable_averages = tf.train.ExponentialMovingAverage(
                        0.9999, global_step
                    )
                    variable_averages_op = variable_averages.apply(
                        tf.trainable_variables()
                    )
                else:
                    variable_averages_op = tf.no_op(name="noop")

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                # Compute Loss (for TRAIN mode)
                loss_dict = mean_cross_entropy_center_loss(
                    logits,
                    prelogits,
                    labels,
                    self.n_classes,
                    alpha=self.alpha,
                    factor=self.factor,
                )

                self.loss = loss_dict["loss"]
                centers = loss_dict["centers"]

                # Compute the moving average of all individual losses and the total loss.
                loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
                loss_averages_op = loss_averages.apply(
                    tf.get_collection(tf.GraphKeys.LOSSES)
                )

                for l in tf.get_collection(tf.GraphKeys.LOSSES):
                    tf.summary.scalar(l.op.name, loss_averages.average(l))

                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(
                        self.extra_checkpoint["checkpoint_path"],
                        self.extra_checkpoint["scopes"],
                    )

                train_op = self.optimize_loss(
                    self.loss,
                    global_step,
                    self.optimize_loss_learning_rate,
                    optimizer,
                    update_ops=[variable_averages_op] + update_ops,
                )

                # Get the moving average saver after optimizer.minimize is called
                if self.apply_moving_averages:
                    self.saver, self.scaffold = moving_average_scaffold(
                        self.optimizer.optimizer
                        if hasattr(self.optimizer, "optimizer")
                        else self.optimizer,
                        config,
                    )
                else:
                    self.saver, self.scaffold = None, None

                train_op = tf.group(
                    train_op, centers, variable_averages_op, loss_averages_op
                )
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, train_op=train_op
                )

            # Building the training graph for PREDICTION OR VALIDATION
            prelogits, end_points = self.architecture(data, mode=mode)
            logits = append_logits(prelogits, n_classes)
            end_points["logits"] = logits

            if self.embedding_validation:
                # Compute the embeddings
                embeddings = tf.nn.l2_normalize(prelogits, 1)
                predictions = {"embeddings": embeddings, "key": key}
            else:
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
                    "key": key,
                }
            end_points.update(predictions)
            self.end_points = end_points

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # IF Validation
            loss_dict = mean_cross_entropy_center_loss(
                logits,
                prelogits,
                labels,
                self.n_classes,
                alpha=self.alpha,
                factor=self.factor,
            )
            self.loss = loss_dict["loss"]

            if self.embedding_validation:
                predictions_op = predict_using_tensors(
                    predictions["embeddings"], labels, num=validation_batch_size
                )
                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions_op
                    )
                }
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops
                )

            else:
                # Add evaluation metrics (for EVAL mode)
                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"]
                    )
                }
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops
                )

        super(LogitsCenterLoss, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config
        )


def moving_average_scaffold(optimizer, config):
    max_to_keep = 5 if config is None else config.keep_checkpoint_max
    keep_checkpoint_every_n_hours = (
        10000.0 if config is None else config.keep_checkpoint_every_n_hours
    )
    saver = optimizer.swapping_saver(
        max_to_keep=max_to_keep,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
    )
    scaffold = tf.train.Scaffold(saver=saver)
    return saver, scaffold
