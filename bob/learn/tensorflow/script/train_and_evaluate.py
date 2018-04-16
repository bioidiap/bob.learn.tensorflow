#!/usr/bin/env python
"""Trains and evaluates a network using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import tensorflow as tf
from bob.learn.tensorflow.utils.hooks import EarlyStopException
import logging
import click
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)

logger = logging.getLogger(__name__)


@click.command(entry_point_group='bob.learn.tensorflow.config',
               cls=ConfigCommand)
@click.option('--estimator', '-e', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.estimator')
@click.option('--train-spec', '-it', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.trainspec')
@click.option('--eval-spec', '-ie', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.evalspec')
@click.option('--exit-ok-exceptions', cls=ResourceOption, multiple=True,
              default=(EarlyStopException,), show_default=True,
              entry_point_group='bob.learn.tensorflow.exception')
@verbosity_option(cls=ResourceOption)
def train_and_evaluate(estimator, train_spec, eval_spec, exit_ok_exceptions,
                       **kwargs):
    """Trains and evaluates a network using Tensorflow estimators.

    This script calls the estimator.train_and_evaluate function. Please see:
    https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
    https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec
    https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec
    for more details.

    \b
    Parameters
    ----------
    estimator : tf.estimator.Estimator
        The estimator that will be trained. Can be a
        ``bob.learn.tensorflow.estimator`` entry point or a path to a Python
        file which contains a variable named `estimator`.
    train_spec : tf.estimator.TrainSpec
        See :any:`tf.estimator.Estimator.train_and_evaluate`.
    eval_spec : tf.estimator.EvalSpec
        See :any:`tf.estimator.Estimator.train_and_evaluate`.
    exit_ok_exceptions : [Exception], optional
        A list of exceptions to exit properly if they occur. If nothing is
        provided, the EarlyStopException is handled by default.
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
    logger.debug('train_spec: %s', train_spec)
    logger.debug('eval_spec: %s', eval_spec)
    logger.debug('exit_ok_exceptions: %s', exit_ok_exceptions)
    logger.debug('kwargs: %s', kwargs)

    # Train and evaluate
    try:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    except exit_ok_exceptions as e:
        logger.exception(e)
        return
