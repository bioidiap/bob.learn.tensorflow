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
from bob.bio.base import is_argument_available

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
         'function. If the ``input_fn`` accepts a ``cache_only`` argument, '
         'it will be given as True.')
@click.option(
    '--mode',
    cls=ResourceOption,
    default=tf.estimator.ModeKeys.TRAIN,
    show_default=True,
    type=click.Choice((tf.estimator.ModeKeys.TRAIN,
                       tf.estimator.ModeKeys.EVAL,
                       tf.estimator.ModeKeys.PREDICT)),
    help='mode value to be given to the input_fn.')
@verbosity_option(cls=ResourceOption)
def cache_dataset(input_fn, mode, **kwargs):
    """Trains networks using Tensorflow estimators."""
    log_parameters(logger)

    kwargs = {}
    if is_argument_available('cache_only', input_fn):
        kwargs['cache_only'] = True
        logger.info("cache_only as True will be passed to input_fn.")

    # call the input function manually
    with tf.Session() as sess:
        data = input_fn(mode, **kwargs)
        if isinstance(data, tf.data.Dataset):
            iterator = data.make_initializable_iterator()
            data = iterator.get_next()
            sess.run(iterator.initializer)
        sess.run(tf.initializers.global_variables())
        try:
            while True:
                sess.run(data)
        except tf.errors.OutOfRangeError:
            click.echo("Finished reading the dataset.")
            return
