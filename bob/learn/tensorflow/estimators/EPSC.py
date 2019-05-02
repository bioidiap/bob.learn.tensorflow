# vim: set fileencoding=utf-8 :
# @author: Amir Mohammadi <amir.mohammadi@idiap.ch>

from . import check_features, get_trainable_variables
from .Logits import moving_average_scaffold
from ..network.utils import append_logits
from ..utils import predict_using_tensors
from ..loss.epsc import epsc_metric, siamese_loss
from tensorflow.python.estimator import estimator
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class EPSCBase:
    """A base class for EPSC based estimators"""

    def _get_loss(self, bio_logits, pad_logits, bio_labels, pad_labels, mode):
        main_loss = self.loss_op(
            bio_logits=bio_logits,
            pad_logits=pad_logits,
            bio_labels=bio_labels,
            pad_labels=pad_labels,
        )
        total_loss = main_loss

        if self.add_regularization_losses:

            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES
            )
            regularization_losses = [
                tf.cast(l, main_loss.dtype) for l in regularization_losses
            ]

            regularization_losses = tf.add_n(
                regularization_losses, name="regularization_losses"
            )
            tf.summary.scalar("regularization_losses", regularization_losses)

            total_loss = tf.add_n([main_loss, regularization_losses], name="total_loss")

        if self.vat_loss is not None:
            vat_loss = self.vat_loss(
                self.end_points["features"],
                self.end_points["Logits/PAD"],
                self.pad_architecture,
                mode,
            )
            total_loss = tf.add_n([main_loss, vat_loss], name="total_loss")

        return total_loss


class EPSCLogits(EPSCBase, estimator.Estimator):
    """An logits estimator for epsc problems"""

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
        extra_checkpoint=None,
        apply_moving_averages=True,
        add_histograms="train",
        add_regularization_losses=True,
        vat_loss=None,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
    ):

        self.architecture = architecture
        self.n_classes = n_classes
        self.loss_op = loss_op
        self.loss = None
        self.embedding_validation = embedding_validation
        self.extra_checkpoint = extra_checkpoint
        self.add_regularization_losses = add_regularization_losses
        self.apply_moving_averages = apply_moving_averages
        self.vat_loss = vat_loss
        self.optimize_loss = optimize_loss
        self.optimize_loss_learning_rate = optimize_loss_learning_rate

        if apply_moving_averages and isinstance(optimizer, tf.train.Optimizer):
            logger.info(
                "Encapsulating the optimizer with " "the MovingAverageOptimizer"
            )
            optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

        self.optimizer = optimizer

        def _model_fn(features, labels, mode):

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

            name = "Logits/Bio"
            bio_logits = append_logits(
                prelogits, n_classes, trainable_variables=trainable_variables, name=name
            )
            end_points[name] = bio_logits

            name = "Logits/PAD"
            pad_logits = append_logits(
                prelogits, 2, trainable_variables=trainable_variables, name=name
            )
            end_points[name] = pad_logits

            self.end_points = end_points

            # for vat_loss
            self.end_points["features"] = data

            def pad_architecture(features, mode, reuse):
                prelogits, end_points = self.architecture(
                    features,
                    mode=mode,
                    trainable_variables=trainable_variables,
                    reuse=reuse,
                )
                pad_logits = append_logits(
                    prelogits,
                    2,
                    reuse=reuse,
                    trainable_variables=trainable_variables,
                    name="Logits/PAD",
                )
                return pad_logits, end_points

            self.pad_architecture = pad_architecture

            if self.embedding_validation and mode != tf.estimator.ModeKeys.TRAIN:

                # Compute the embeddings
                embeddings = tf.nn.l2_normalize(prelogits, 1)
                predictions = {"embeddings": embeddings}
            else:
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "bio_classes": tf.argmax(input=bio_logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT
                    # and by the `logging_hook`.
                    "bio_probabilities": tf.nn.softmax(
                        bio_logits, name="bio_softmax_tensor"
                    ),
                }

            predictions.update(
                {
                    "pad_classes": tf.argmax(input=pad_logits, axis=1),
                    "pad_probabilities": tf.nn.softmax(
                        pad_logits, name="pad_softmax_tensor"
                    ),
                    "key": key,
                }
            )

            # add predictions to end_points
            self.end_points.update(predictions)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            bio_labels = labels["bio"]
            pad_labels = labels["pad"]

            if self.embedding_validation and mode != tf.estimator.ModeKeys.TRAIN:
                bio_predictions_op = predict_using_tensors(
                    predictions["embeddings"], bio_labels, num=validation_batch_size
                )
            else:
                bio_predictions_op = predictions["bio_classes"]

            pad_predictions_op = predictions["pad_classes"]

            metrics = {
                "bio_accuracy": tf.metrics.accuracy(
                    labels=bio_labels, predictions=bio_predictions_op
                ),
                "pad_accuracy": tf.metrics.accuracy(
                    labels=pad_labels, predictions=pad_predictions_op
                ),
            }

            if mode == tf.estimator.ModeKeys.EVAL:
                self.loss = self._get_loss(
                    bio_logits, pad_logits, bio_labels, pad_labels, mode=mode
                )
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=self.loss,
                    train_op=None,
                    eval_metric_ops=metrics,
                )

            # restore the model from an extra_checkpoint
            if self.extra_checkpoint is not None:
                if "Logits/" not in self.extra_checkpoint["scopes"]:
                    logger.warning(
                        '"Logits/" (which are automatically added by this '
                        "Logits class are not in the scopes of "
                        "extra_checkpoint). Did you mean to restore the "
                        "Logits variables as well?"
                    )

                logger.info(
                    "Restoring model from %s in scopes %s",
                    self.extra_checkpoint["checkpoint_path"],
                    self.extra_checkpoint["scopes"],
                )
                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=self.extra_checkpoint["checkpoint_path"],
                    assignment_map=self.extra_checkpoint["scopes"],
                )

            # Calculate Loss
            self.loss = self._get_loss(
                bio_logits, pad_logits, bio_labels, pad_labels, mode=mode
            )

            # Compute the moving average of all individual losses and the total
            # loss.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
            loss_averages_op = loss_averages.apply(
                tf.get_collection(tf.GraphKeys.LOSSES)
            )
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, loss_averages_op)

            with tf.name_scope("train"):
                train_op = self.optimize_loss(
                    loss=self.loss,
                    global_step=tf.train.get_or_create_global_step(),
                    optimizer=self.optimizer,
                    learning_rate=self.optimize_loss_learning_rate,
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
                    tf.summary.scalar("bio_accuracy", metrics["bio_accuracy"][1])
                    tf.summary.scalar("pad_accuracy", metrics["pad_accuracy"][1])
                    for l in tf.get_collection(tf.GraphKeys.LOSSES):
                        tf.summary.scalar(
                            l.op.name + "_averaged", loss_averages.average(l)
                        )

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

        super().__init__(model_fn=_model_fn, model_dir=model_dir, config=config)


class EPSCSiamese(EPSCBase, estimator.Estimator):
    """An siamese estimator for epsc problems"""

    def __init__(
        self,
        architecture,
        optimizer,
        loss_op=siamese_loss,
        config=None,
        model_dir="",
        validation_batch_size=None,
        extra_checkpoint=None,
        apply_moving_averages=True,
        add_histograms="train",
        add_regularization_losses=True,
        vat_loss=None,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
    ):

        self.architecture = architecture
        self.loss_op = loss_op
        self.loss = None
        self.extra_checkpoint = extra_checkpoint
        self.add_regularization_losses = add_regularization_losses
        self.apply_moving_averages = apply_moving_averages
        self.vat_loss = vat_loss
        self.optimize_loss = optimize_loss
        self.optimize_loss_learning_rate = optimize_loss_learning_rate

        if self.apply_moving_averages and isinstance(optimizer, tf.train.Optimizer):
            logger.info(
                "Encapsulating the optimizer with " "the MovingAverageOptimizer"
            )
            optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

        self.optimizer = optimizer

        def _model_fn(features, labels, mode):

            if mode != tf.estimator.ModeKeys.TRAIN:
                check_features(features)
                data = features["data"]
                key = features["key"]
            else:
                if "left" not in features or "right" not in features:
                    raise ValueError(
                        "The input features needs to be a dictionary "
                        "with the keys `left` and `right`"
                    )
                data_right = features["right"]["data"]
                labels_right = labels["right"]
                data = features["left"]["data"]
                labels = labels_left = labels["left"]

            # Checking if we have some variables/scope that we may want to shut
            # down
            trainable_variables = get_trainable_variables(
                self.extra_checkpoint, mode=mode
            )

            prelogits, end_points = self.architecture(
                data, mode=mode, trainable_variables=trainable_variables
            )

            self.end_points = end_points

            predictions = dict(
                bio_embeddings=tf.nn.l2_normalize(prelogits, 1),
                pad_probabilities=tf.math.exp(-tf.norm(prelogits, ord=2, axis=-1)),
            )

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions["key"] = key

            # add predictions to end_points
            self.end_points.update(predictions)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            metrics = None
            if mode != tf.estimator.ModeKeys.TRAIN:
                assert validation_batch_size is not None
                bio_labels = labels["bio"]
                pad_labels = labels["pad"]

                metrics = epsc_metric(
                    predictions["bio_embeddings"],
                    predictions["pad_probabilities"],
                    bio_labels,
                    pad_labels,
                    validation_batch_size,
                )

            if mode == tf.estimator.ModeKeys.EVAL:
                self.loss = tf.reduce_mean(0)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=self.loss,
                    train_op=None,
                    eval_metric_ops=metrics,
                )

            # now that we are in TRAIN mode, build the right graph too
            prelogits_left = prelogits
            prelogits_right, _ = self.architecture(
                data_right,
                mode=mode,
                reuse=True,
                trainable_variables=trainable_variables,
            )

            bio_logits = {"left": prelogits_left, "right": prelogits_right}
            pad_logits = bio_logits

            bio_labels = {"left": labels_left["bio"], "right": labels_right["bio"]}

            pad_labels = {"left": labels_left["pad"], "right": labels_right["pad"]}

            # restore the model from an extra_checkpoint
            if self.extra_checkpoint is not None:
                logger.info(
                    "Restoring model from %s in scopes %s",
                    self.extra_checkpoint["checkpoint_path"],
                    self.extra_checkpoint["scopes"],
                )
                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=self.extra_checkpoint["checkpoint_path"],
                    assignment_map=self.extra_checkpoint["scopes"],
                )

            global_step = tf.train.get_or_create_global_step()

            # Some layer like tf.layers.batch_norm need this:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops), tf.name_scope("train"):

                # Calculate Loss
                self.loss = self._get_loss(
                    bio_logits, pad_logits, bio_labels, pad_labels, mode=mode
                )

                # Compute the moving average of all individual losses
                # and the total loss.
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
                if apply_moving_averages:
                    self.saver, self.scaffold = moving_average_scaffold(
                        self.optimizer.optimizer
                        if hasattr(self.optimizer, "optimizer")
                        else self.optimizer,
                        config,
                    )
                else:
                    self.saver, self.scaffold = None, None

            # Log moving average of losses
            with tf.name_scope("train_metrics"):
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

        super().__init__(model_fn=_model_fn, model_dir=model_dir, config=config)
