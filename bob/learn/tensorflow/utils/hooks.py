from datetime import datetime
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
import logging
import numpy as np
import tensorflow as tf
import time

logger = logging.getLogger(__name__)


class TensorSummary(tf.estimator.SessionRunHook):
    """Adds the given (scalar) tensors to tensorboard summaries"""

    def __init__(self, tensors, tensor_names=None, **kwargs):
        super().__init__(**kwargs)
        self.tensors = list(tensors)
        if tensor_names is None:
            tensor_names = [t.name for t in self.tensors]
        self.tensor_names = list(tensor_names)

    def begin(self):
        for name, tensor in zip(self.tensor_names, self.tensors):
            tf.summary.scalar(name, tensor)


class LoggerHook(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, loss, batch_size, log_frequency):
        self.loss = loss
        self.batch_size = batch_size
        self.log_frequency = log_frequency

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        if self._step % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = self.log_frequency * self.batch_size / duration
            sec_per_batch = float(duration / self.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), self._step, loss_value,
                                examples_per_sec, sec_per_batch))


class LoggerHookEstimator(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, estimator, batch_size, log_frequency):
        self.estimator = estimator
        self.batch_size = batch_size
        self.log_frequency = log_frequency

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        # Asks for loss value.
        return tf.train.SessionRunArgs(self.estimator.loss)

    def after_run(self, run_context, run_values):
        if self._step % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = self.log_frequency * self.batch_size / duration
            sec_per_batch = float(duration / self.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), self._step, loss_value,
                                examples_per_sec, sec_per_batch))


class EarlyStopException(Exception):
    pass


class EarlyStopping(tf.estimator.SessionRunHook):
    """Stop training when a monitored quantity has stopped improving.
    Based on Keras's EarlyStopping callback:
    https://keras.io/callbacks/#earlystopping
    The original implementation worked for epochs. Currently there is no way
    to know the epoch count in estimator training. Hence, the criteria is done
    using steps instead of epochs.

    Parameters
    ----------
    monitor
        quantity to be monitored.
    min_delta
        minimum change in the monitored quantity to qualify as an improvement,
        i.e. an absolute change of less than min_delta, will count as no
        improvement.
    patience
        number of steps with no improvement after which training will be
        stopped. Please use large patience values since this hook is
        implemented using steps instead of epochs compared to the equivalent
        one in Keras.
    mode
        one of {auto, min, max}. In `min` mode, training will stop when the
        quantity monitored has stopped decreasing; in `max` mode it will stop
        when the quantity monitored has stopped increasing; in `auto` mode, the
        direction is automatically inferred from the name of the monitored
        quantity.
    """

    def __init__(self,
                 monitor='accuracy/value',
                 min_delta=0,
                 patience=0,
                 mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            logger.warn('EarlyStopping mode %s is unknown, '
                        'fallback to auto mode.' % mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.global_step_of_best = 0

    def begin(self):
        self.values = []
        if isinstance(self.monitor, str):
            self.monitor = _as_graph_element(self.monitor)
        else:
            self.monitor = _as_graph_element(self.monitor.name)
        self.global_step_tensor = tf.train.get_global_step()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.monitor, self.global_step_tensor])

    def after_run(self, run_context, run_values):
        monitor, global_step = run_values.results
        self.values.append(monitor)
        # global step does not change during evaluation so keeping one of them
        # is enough.
        self.global_step_value = global_step

    def _should_stop(self):
        current = np.mean(self.values)
        logger.info(
            '%s is currently at %f (at step of %d) and the best value was %f '
            '(at step of %d)', self.monitor.name, current,
            self.global_step_value, self.best, self.global_step_of_best)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.global_step_of_best = self.global_step_value
        else:
            if self.wait >= self.patience:
                message = 'Early stopping happened with {} at best of ' \
                    '{} (at step {}) and current of {} (at step {})'.format(
                        self.monitor.name, self.best, self.global_step_of_best,
                        current, self.global_step_value)
                logger.info(message)
                raise EarlyStopException(message)
            self.wait += 1

    def end(self, session):
        self._should_stop()
