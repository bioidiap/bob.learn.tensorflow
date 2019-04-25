#!/usr/bin/env python
"""Trains networks using Keras Models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import click
import json
import logging
import os
import tensorflow as tf
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--model',
    '-m',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.model',
    help='The keras model that will be trained.')
@click.option(
    '--train-input-fn',
    '-i',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.input_fn',
    help='A function that will return the training data as a tf.data.Dataset '
    'or tf.data.Iterator. This will be given as `x` to '
    'tf.keras.Model.fit.')
@click.option(
    '--epochs',
    '-e',
    default=1,
    type=click.types.INT,
    cls=ResourceOption,
    help='Number of epochs to train model. See '
    'tf.keras.Model.fit.')
@click.option(
    '--callbacks',
    cls=ResourceOption,
    multiple=True,
    entry_point_group='bob.learn.tensorflow.callback',
    help='List of tf.keras.callbacks. Used for callbacks '
    'inside the training loop.')
@click.option(
    '--eval-input-fn',
    '-i',
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.input_fn',
    help='A function that will return the validation data as a tf.data.Dataset'
    ' or tf.data.Iterator. This will be given as `validation_data` to '
    'tf.keras.Model.fit.')
@click.option(
    '--class-weight',
    '-c',
    cls=ResourceOption,
    help='See tf.keras.Model.fit.')
@click.option(
    '--initial-epoch',
    default=0,
    type=click.types.INT,
    cls=ResourceOption,
    help='See tf.keras.Model.fit.')
@click.option(
    '--steps-per-epoch',
    type=click.types.INT,
    cls=ResourceOption,
    help='See tf.keras.Model.fit.')
@click.option(
    '--validation-steps',
    type=click.types.INT,
    cls=ResourceOption,
    help='See tf.keras.Model.fit.')
@verbosity_option(cls=ResourceOption)
def fit(model, train_input_fn, epochs, verbose, callbacks, eval_input_fn,
        class_weight, initial_epoch, steps_per_epoch, validation_steps,
        **kwargs):
    """Trains networks using Keras models."""
    log_parameters(logger)

    # Train
    save_callback = [c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint)]
    model_dir = None
    if save_callback:
        model_dir = save_callback[0].filepath
        logger.info("Training a model in %s", model_dir)
    history = model.fit(
        x=train_input_fn(),
        epochs=epochs,
        verbose=max(verbose, 2),
        callbacks=list(callbacks) if callbacks else None,
        validation_data=None if eval_input_fn is None else eval_input_fn(),
        class_weight=class_weight,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    click.echo(history.history)
    if model_dir is not None:
        with open(os.path.join(model_dir, 'keras_fit_history.json'), 'w') as f:
            json.dump(history.history, f)
