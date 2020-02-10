from bob.extension.config import load as read_config_files
from bob.io.base.test_utils import datafile
from bob.learn.tensorflow.estimators import Logits
from bob.learn.tensorflow.loss.BaseLoss import mean_cross_entropy_loss
from bob.learn.tensorflow.utils.hooks import EarlyStopping, EarlyStopException
import nose
import tensorflow as tf
import shutil
from nose.plugins.attrib import attr


# @nose.tools.raises(EarlyStopException)
# @attr('slow')
# def test_early_stopping_linear_classifier():
#     config = read_config_files([
#         datafile('mnist_input_fn.py', __name__),
#         datafile('mnist_estimator.py', __name__),
#     ])
#     estimator = config.estimator
#     train_input_fn = config.train_input_fn
#     eval_input_fn = config.eval_input_fn

#     hooks = [
#         EarlyStopping(
#             'linear/metrics/accuracy/total', min_delta=0.001, patience=1),
#     ]

#     train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
#     eval_spec = tf.estimator.EvalSpec(
#         input_fn=eval_input_fn, hooks=hooks, throttle_secs=2, steps=10)

#     try:
#         tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
#     finally:
#         shutil.rmtree(estimator.model_dir)


# @nose.tools.raises(EarlyStopException)
# @attr('slow')
# def test_early_stopping_logit_trainer():
#     config = read_config_files([
#         datafile('mnist_input_fn.py', __name__),
#     ])
#     train_input_fn = config.train_input_fn
#     eval_input_fn = config.eval_input_fn

#     hooks = [
#         EarlyStopping('accuracy/value', min_delta=0.001, patience=1),
#     ]

#     train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
#     eval_spec = tf.estimator.EvalSpec(
#         input_fn=eval_input_fn, hooks=hooks, throttle_secs=2, steps=10)

#     def architecture(data, mode, **kwargs):
#         return data, dict()

#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
#     loss_op = mean_cross_entropy_loss

#     estimator = Logits(
#         architecture, optimizer, loss_op, n_classes=10, model_dir=None)

#     try:
#         tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
#     finally:
#         shutil.rmtree(estimator.model_dir)
