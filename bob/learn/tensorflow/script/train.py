#!/usr/bin/env python
"""Trains networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import click
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--estimator',
    '-e',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.estimator',
    help='The estimator that will be trained.')
@click.option(
    '--train-input-fn',
    '-i',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.input_fn',
    help='The ``input_fn`` that will be given to '
    ':any:`tf.estimator.Estimator.train`.')
@click.option(
    '--hooks',
    cls=ResourceOption,
    multiple=True,
    entry_point_group='bob.learn.tensorflow.hook',
    help='List of SessionRunHook subclass instances. Used for callbacks '
    'inside the training loop.')
@click.option(
    '--steps',
    '-s',
    cls=ResourceOption,
    type=click.types.INT,
    help='Number of steps for which to train model. See '
    ':any:`tf.estimator.Estimator.train`.')
@click.option(
    '--max-steps',
    '-m',
    cls=ResourceOption,
    type=click.types.INT,
    help='Number of total steps for which to train model. See '
    ':any:`tf.estimator.Estimator.train`.')
@verbosity_option(cls=ResourceOption)
def train(estimator, train_input_fn, hooks, steps, max_steps, **kwargs):
    """Trains networks using Tensorflow estimators."""
    log_parameters(logger)

    # Train
    logger.info("Training a model in %s", estimator.model_dir)
    estimator.train(
        input_fn=train_input_fn, hooks=hooks, steps=steps, max_steps=max_steps)
