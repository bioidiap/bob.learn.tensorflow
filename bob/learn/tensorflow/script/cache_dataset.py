#!/usr/bin/env python
"""Trains networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import click
import tensorflow as tf
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--input-fn',
    '-i',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.input_fn',
    help='The ``input_fn`` that will return the features and labels. '
         'You should call the dataset.cache(...) yourself in the input '
         'function.')
@click.option(
    '--mode',
    cls=ResourceOption,
    default='train',
    show_default=True,
    help='One of the tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT} values to be '
    'given to the input_fn.')
@verbosity_option(cls=ResourceOption)
def cache_dataset(input_fn, mode, **kwargs):
    """Trains networks using Tensorflow estimators."""
    log_parameters(logger)

    # call the input function manually
    with tf.Session() as sess:
        data = input_fn(mode)
        while True:
            sess.run(data)
