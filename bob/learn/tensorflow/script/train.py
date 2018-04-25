#!/usr/bin/env python
"""Trains networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import click
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)

logger = logging.getLogger(__name__)


@click.command(entry_point_group='bob.learn.tensorflow.config',
               cls=ConfigCommand)
@click.option('--estimator', '-e', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.estimator')
@click.option('--train-input-fn', '-i', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.input_fn')
@click.option('--hooks', cls=ResourceOption, multiple=True,
              entry_point_group='bob.learn.tensorflow.hook')
@click.option('--steps', '-s', cls=ResourceOption, type=click.types.INT)
@click.option('--max-steps', '-m', cls=ResourceOption, type=click.types.INT)
@verbosity_option(cls=ResourceOption)
def train(estimator, train_input_fn, hooks, steps, max_steps, **kwargs):
    """Trains networks using Tensorflow estimators.

    \b
    Parameters
    ----------
    estimator : tf.estimator.Estimator
        The estimator that will be trained. Can be a
        ``bob.learn.tensorflow.estimator`` entry point or a path to a Python
        file which contains a variable named `estimator`.
    train_input_fn : callable
        The ``input_fn`` that will be given to
        :any:`tf.estimator.Estimator.train`. Can be a
        ``bob.learn.tensorflow.input_fn`` entry point or a path to a Python
        file which contains a variable named `train_input_fn`.
    hooks : [tf.train.SessionRunHook], optional
        List of SessionRunHook subclass instances. Used for callbacks inside
        the training loop. Can be a ``bob.learn.tensorflow.hook`` entry point
        or a path to a Python file which contains a variable named `hooks`.
    steps : int, optional
        Number of steps for which to train model. See
        :any:`tf.estimator.Estimator.train`.
    max_steps : int, optional
        Number of total steps for which to train model. See
        :any:`tf.estimator.Estimator.train`.
    verbose : int, optional
        Increases verbosity (see help for --verbose).

    \b
    [CONFIG]...            Configuration files. It is possible to pass one or
                           several Python files (or names of
                           ``bob.learn.tensorflow.config`` entry points or
                           module names) which contain the parameters listed
                           above as Python variables. The options through the
                           command-line (see below) will override the values of
                           configuration files.
    """
    logger.debug('estimator: %s', estimator)
    logger.debug('train_input_fn: %s', train_input_fn)
    logger.debug('hooks: %s', hooks)
    logger.debug('steps: %s', steps)
    logger.debug('max_steps: %s', max_steps)
    logger.debug('kwargs: %s', kwargs)

    # Train
    logger.info("Training a model in %s", estimator.model_dir)
    estimator.train(
        input_fn=train_input_fn, hooks=hooks, steps=steps, max_steps=max_steps)
