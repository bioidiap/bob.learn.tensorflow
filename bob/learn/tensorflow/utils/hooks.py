import tensorflow as tf
import time
from datetime import datetime
import logging
import numpy as np
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element


class LoggerHook(tf.train.SessionRunHook):
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


class LoggerHookEstimator(tf.train.SessionRunHook):
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


class EarlyStopping(tf.train.SessionRunHook):
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
                 monitor='accuracy',
                 min_delta=0,
                 patience=0,
                 mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
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

    def begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.monitor = _as_graph_element(self.monitor)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.monitor)

    def after_run(self, run_context, run_values):
        current = run_values.results
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                run_context.request_stop()
                print('Early stopping happened with {} at best of {} and '
                      'current of {}'.format(
                          self.monitor, self.best, current))
            self.wait += 1
