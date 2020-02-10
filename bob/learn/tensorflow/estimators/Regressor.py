from . import check_features, get_trainable_variables
from .Logits import moving_average_scaffold
from bob.learn.tensorflow.network.utils import append_logits
from tensorflow.python.estimator import estimator
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class Regressor(estimator.Estimator):
    """An estimator for regression problems"""

    def __init__(
        self,
        architecture,
        optimizer=tf.train.AdamOptimizer(),
        loss_op=tf.losses.mean_squared_error,
        label_dimension=1,
        config=None,
        model_dir=None,
        apply_moving_averages=True,
        add_regularization_losses=True,
        extra_checkpoint=None,
        add_histograms=None,
        optimize_loss=tf.contrib.layers.optimize_loss,
        optimize_loss_learning_rate=None,
        architecture_has_logits=False,
    ):
        self.architecture = architecture
        self.label_dimension = label_dimension
        self.loss_op = loss_op
        self.add_regularization_losses = add_regularization_losses
        self.apply_moving_averages = apply_moving_averages

        if self.apply_moving_averages and isinstance(optimizer, tf.train.Optimizer):
            logger.info(
                "Encapsulating the optimizer with " "the MovingAverageOptimizer"
            )
            optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

        self.optimizer = optimizer
        self.optimize_loss = optimize_loss
        self.optimize_loss_learning_rate = optimize_loss_learning_rate

        def _model_fn(features, labels, mode, config):

            check_features(features)
            data = features["data"]
            key = features["key"]

            # Checking if we have some variables/scope that we may want to shut
            # down
            trainable_variables = get_trainable_variables(extra_checkpoint, mode=mode)
            prelogits, end_points = self.architecture(
                data, mode=mode, trainable_variables=trainable_variables
            )
            if architecture_has_logits:
                logits, prelogits = prelogits, end_points["prelogits"]
            else:
                logits = append_logits(
                    prelogits, label_dimension, trainable_variables=trainable_variables
                )

            predictions = {"predictions": logits, "key": key}

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # in PREDICT mode logits rank must be 2 but in EVAL and TRAIN the
            # rank should be 1 for the loss function!
            predictions["predictions"] = tf.squeeze(logits)

            predictions_op = predictions["predictions"]

            # Calculate root mean squared error
            rmse = tf.metrics.root_mean_squared_error(labels, predictions_op)
            metrics = {"rmse": rmse}

            if mode == tf.estimator.ModeKeys.EVAL:
                self.loss = self._get_loss(predictions=predictions_op, labels=labels)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=self.loss,
                    train_op=None,
                    eval_metric_ops=metrics,
                )

            # restore the model from an extra_checkpoint
            if extra_checkpoint is not None:
                if "Logits/" not in extra_checkpoint["scopes"]:
                    logger.warning(
                        '"Logits/" (which are automatically added by this '
                        "Regressor class are not in the scopes of "
                        "extra_checkpoint). Did you mean to restore the "
                        "Logits variables as well?"
                    )
                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=extra_checkpoint["checkpoint_path"],
                    assignment_map=extra_checkpoint["scopes"],
                )

            # Calculate Loss
            self.loss = self._get_loss(predictions=predictions_op, labels=labels)

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
            if self.apply_moving_averages:
                self.saver, self.scaffold = moving_average_scaffold(
                    self.optimizer.optimizer
                    if hasattr(self.optimizer, "optimizer")
                    else self.optimizer,
                    config,
                )
            else:
                self.saver, self.scaffold = None, None

            # Log rmse and loss
            with tf.name_scope("train_metrics"):
                tf.summary.scalar("rmse", rmse[1])
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

        super(Regressor, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config
        )

    def _get_loss(self, predictions, labels):
        main_loss = self.loss_op(predictions=predictions, labels=labels)
        if not self.add_regularization_losses:
            return main_loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_losses = [
            tf.cast(l, main_loss.dtype) for l in regularization_losses
        ]
        total_loss = tf.add_n([main_loss] + regularization_losses, name="total_loss")
        return total_loss
