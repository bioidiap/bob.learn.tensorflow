import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class MovingAverageOptimizer:

    """Creates a callable that can be given to bob.learn.tensorflow.estimators

    This class is useful when you want to have a learning_rate_decay_fn **and** a moving
    average optimizer **and** use bob.learn.tensorflow.estimators

    Attributes
    ----------
    optimizer : object
        A tf.train.Optimizer that is created and wrapped with
        tf.contrib.opt.MovingAverageOptimizer.

    Example
    -------
    >>> import tensorflow as tf
    >>> from bob.learn.tensorflow.estimators import MovingAverageOptimizer
    >>> optimizer = MovingAverageOptimizer("adam")
    >>> actual_optimizer = optimizer(lr=1e-3)
    >>> isinstance(actual_optimizer, tf.train.Optimizer)
    True
    >>> actual_optimizer is optimizer.optimizer
    True
    """

    def __init__(self, optimizer, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(optimizer, str), optimizer
        self._optimizer = optimizer

    def __call__(self, lr):
        logger.info("Encapsulating the optimizer with the MovingAverageOptimizer")

        if self._optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif self._optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

        return self.optimizer


def learning_rate_decay_fn(
    learning_rate, global_step, decay_steps, decay_rate, staircase=False
):
    """A simple learning_rate_decay_fn.

    To use it with ``tf.contrib.layer.optimize_loss``:

    >>> from bob.learn.tensorflow.estimators import learning_rate_decay_fn
    >>> from functools import partial
    >>> learning_rate_decay_fn = partial(
    ...     learning_rate_decay_fn,
    ...     decay_steps=1000,
    ...     decay_rate=0.9,
    ...     staircase=True,
    ... )
    """
    return tf.train.exponential_decay(
        learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )
