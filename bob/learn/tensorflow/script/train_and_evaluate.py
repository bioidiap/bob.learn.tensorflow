#!/usr/bin/env python
"""Trains and evaluates a network using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from bob.learn.tensorflow.utils.hooks import EarlyStopException
import logging
import click
from bob.extension.scripts.click_helper import (verbosity_option,
                                                ConfigCommand, ResourceOption)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--estimator',
    '-e',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.estimator',
    help='The estimator that will be trained and evaluated.')
@click.option(
    '--train-spec',
    '-it',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.trainspec',
    help='See :any:`tf.estimator.Estimator.train_and_evaluate`.')
@click.option(
    '--eval-spec',
    '-ie',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.evalspec',
    help='See :any:`tf.estimator.Estimator.train_and_evaluate`.')
@click.option(
    '--exit-ok-exceptions',
    cls=ResourceOption,
    multiple=True,
    default=(EarlyStopException, ),
    show_default=True,
    entry_point_group='bob.learn.tensorflow.exception',
    help='A list of exceptions to exit properly if they occur. If nothing is '
    'provided, the EarlyStopException is handled by default.')
@verbosity_option(cls=ResourceOption)
def train_and_evaluate(estimator, train_spec, eval_spec, exit_ok_exceptions,
                       **kwargs):
    """Trains and evaluates a network using Tensorflow estimators.

    This script calls the estimator.train_and_evaluate function. Please see:
    https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
    https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec
    https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec
    for more details.
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
