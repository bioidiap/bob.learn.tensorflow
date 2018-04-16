#!/usr/bin/env python
"""Evaluates networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import logging
import os
import time
import six
import sys
import tensorflow as tf
from ..utils.eval import get_global_step
import click
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)

logger = logging.getLogger(__name__)


@click.command(entry_point_group='bob.learn.tensorflow.config',
               cls=ConfigCommand)
@click.option('--estimator', '-e', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.estimator')
@click.option('--eval-input-fn', '-i', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.input_fn')
@click.option('--hooks', cls=ResourceOption, multiple=True,
              entry_point_group='bob.learn.tensorflow.hook')
@click.option('--run-once', cls=ResourceOption, default=False,
              show_default=True)
@click.option('--eval-interval-secs', cls=ResourceOption, type=click.types.INT,
              default=60, show_default=True)
@click.option('--name', cls=ResourceOption)
@verbosity_option(cls=ResourceOption)
def eval(estimator, eval_input_fn, hooks, run_once, eval_interval_secs, name,
         **kwargs):
    """Evaluates networks using Tensorflow estimators.

    \b
    Parameters
    ----------
    estimator : tf.estimator.Estimator
        The estimator that will be trained. Can be a
        ``bob.learn.tensorflow.estimator`` entry point or a path to a Python
        file which contains a variable named `estimator`.
    eval_input_fn : callable
        The ``input_fn`` that will be given to
        :any:`tf.estimator.Estimator.train`. Can be a
        ``bob.learn.tensorflow.input_fn`` entry point or a path to a Python
        file which contains a variable named `eval_input_fn`.
    hooks : [tf.train.SessionRunHook], optional
        List of SessionRunHook subclass instances. Used for callbacks inside
        the training loop. Can be a ``bob.learn.tensorflow.hook`` entry point
        or a path to a Python file which contains a variable named `hooks`.
    run_once : bool, optional
        If given, the model will be evaluated only once.
    eval_interval_secs : int, optional
        The seconds to wait for the next evaluation.
    name : str, optional
        Name of the evaluation
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
    logger.debug('eval_input_fn: %s', eval_input_fn)
    logger.debug('hooks: %s', hooks)
    logger.debug('run_once: %s', run_once)
    logger.debug('eval_interval_secs: %s', eval_interval_secs)
    logger.debug('name: %s', name)
    logger.debug('kwargs: %s', kwargs)

    if name:
        real_name = 'eval_' + name
    else:
        real_name = 'eval'
    evaluated_file = os.path.join(estimator.model_dir, real_name, 'evaluated')
    while True:
        evaluated_steps = []
        if os.path.exists(evaluated_file):
            with open(evaluated_file) as f:
                evaluated_steps = [line.split()[0] for line in f]

        ckpt = tf.train.get_checkpoint_state(estimator.model_dir)
        if (not ckpt) or (not ckpt.model_checkpoint_path):
            time.sleep(eval_interval_secs)
            continue

        for checkpoint_path in ckpt.all_model_checkpoint_paths:
            try:
                global_step = str(get_global_step(checkpoint_path))
            except Exception:
                print('Failed to find global_step for checkpoint_path {}, '
                      'skipping ...'.format(checkpoint_path))
                continue
            if global_step in evaluated_steps:
                continue

            # Evaluate
            evaluations = estimator.evaluate(
                input_fn=eval_input_fn,
                steps=None,
                hooks=hooks,
                checkpoint_path=checkpoint_path,
                name=name,
            )

            str_evaluations = ', '.join(
                '%s = %s' % (k, v)
                for k, v in sorted(six.iteritems(evaluations)))
            print(str_evaluations)
            sys.stdout.flush()
            with open(evaluated_file, 'a') as f:
                f.write('{} {}\n'.format(evaluations['global_step'],
                                         str_evaluations))
        if run_once:
            break
        time.sleep(eval_interval_secs)
