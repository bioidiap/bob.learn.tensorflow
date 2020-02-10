#!/usr/bin/env python
"""Evaluates networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import click
import logging
import os
import shutil
import sys
import tensorflow as tf
import time
from glob import glob
from collections import defaultdict, OrderedDict
from ..utils.eval import get_global_step
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)
from bob.io.base import create_directories_safe

logger = logging.getLogger(__name__)


def copy_one_step(train_dir, global_step, save_dir, fail_on_error=False):
    for path in glob('{}/model.ckpt-{}.*'.format(train_dir, global_step)):
        dst = os.path.join(save_dir, os.path.basename(path))
        if os.path.isfile(dst):
            continue
        try:
            shutil.copy(path, dst)
            logger.info("Copied `%s' over to `%s'", path, dst)
        except OSError:
            logger.warning(
                "Failed to copy `%s' over to `%s'", path, dst,
                exc_info=True)
            if fail_on_error:
                raise


def save_n_best_models(train_dir, save_dir, evaluated_file,
                       keep_n_best_models, sort_by, exceptions=tuple()):
    logger.debug(
        "save_n_best_models was called with %s, %s, %s, %s, %s, %s",
        train_dir, save_dir, evaluated_file, keep_n_best_models, sort_by,
        exceptions)

    create_directories_safe(save_dir)
    evaluated = read_evaluated_file(evaluated_file)

    def _key(x):
        x = x[1][sort_by]
        if 'loss' in sort_by:
            return x
        else:
            return -x

    best_models = OrderedDict(
        sorted(evaluated.items(), key=_key)[:keep_n_best_models])
    logger.info("Best models: %s", best_models)

    # delete the old saved models that are not in top N best anymore
    saved_models = defaultdict(list)
    for path in glob('{}/model.ckpt-*'.format(save_dir)):
        global_step = path.split('model.ckpt-')[1].split('.')[0]
        saved_models[global_step].append(path)

    for global_step, paths in saved_models.items():
        if global_step not in best_models and global_step not in exceptions:
            for path in paths:
                logger.info("Deleting `%s'", path)
                os.remove(path)

    # copy over the best models if not already there
    for global_step in best_models:
        copy_one_step(train_dir, global_step, save_dir)

    # create a checkpoint file indicating to the best existing model:
    # 1. filter non-existing models first
    def _filter(x):
        return len(glob('{}/model.ckpt-{}.*'.format(save_dir, x[0]))) > 0

    best_models = OrderedDict(filter(_filter, best_models.items()))

    # 2. create the checkpoint file
    with open(os.path.join(save_dir, 'checkpoint'), 'wt') as f:
        if not len(best_models):
            return
        the_best_global_step = list(best_models)[0]
        f.write('model_checkpoint_path: "model.ckpt-{}"\n'.format(
            the_best_global_step))
        # reverse the models before saving since the last ones in checkpoints
        # are usually more important. This aligns with the bob tf trim script.
        for i, global_step in enumerate(reversed(best_models)):
            f.write('all_model_checkpoint_paths: "model.ckpt-{}"\n'.format(
                global_step))


def read_evaluated_file(path):
    evaluated = {}
    with open(path) as f:
        for line in f:
            global_step, line = line.split(' ', 1)
            temp = {}
            for k_v in line.strip().split(', '):
                k, v = k_v.split(' = ')
                try:
                    v = float(v)
                except ValueError:  # not all values could be floats
                    pass
                if 'global_step' in k:
                    v = int(v)
                temp[k] = v
            evaluated[global_step] = temp
    return evaluated


def append_evaluated_file(path, evaluations):
    str_evaluations = ', '.join(
        '%s = %s' % (k, v) for k, v in sorted(evaluations.items()))
    with open(path, 'a') as f:
        f.write('{} {}\n'.format(evaluations['global_step'], str_evaluations))
    return str_evaluations


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--estimator',
    '-e',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.estimator',
    help='The estimator that will be evaluated.')
@click.option(
    '--eval-input-fn',
    '-i',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.input_fn',
    help='The ``input_fn`` that will be given to '
    ':any:`tf.estimator.Estimator.eval`.')
@click.option(
    '--hooks',
    cls=ResourceOption,
    multiple=True,
    entry_point_group='bob.learn.tensorflow.hook',
    help='List of SessionRunHook subclass instances. Used for callbacks '
    'inside the evaluation loop.')
@click.option(
    '--run-once',
    cls=ResourceOption,
    default=False,
    show_default=True,
    is_flag=True,
    help='If given, the model will be evaluated only once.')
@click.option(
    '--eval-interval-secs',
    cls=ResourceOption,
    type=click.INT,
    default=60,
    show_default=True,
    help='The seconds to wait for the next evaluation.')
@click.option('--name', cls=ResourceOption, help='Name of the evaluation')
@click.option(
    '--keep-n-best-models',
    '-K',
    type=click.INT,
    cls=ResourceOption,
    default=1,
    show_default=True,
    help='If more than 0, will keep the best N models in the evaluation folder'
)
@click.option(
    '--sort-by',
    cls=ResourceOption,
    default="loss",
    show_default=True,
    help='The metric for sorting the N best models.')
@click.option(
    '--max-wait-intervals',
    cls=ResourceOption,
    type=click.INT,
    default=-1,
    show_default=True,
    help='If given, the maximum number of intervals waiting for new training checkpoint.')
@click.option(
    '--force-re-run',
    is_flag=True,
    default=False,
    help='A debugging flag. Do not use!')
@verbosity_option(cls=ResourceOption)
def eval(estimator, eval_input_fn, hooks, run_once, eval_interval_secs, name,
         keep_n_best_models, sort_by, max_wait_intervals, force_re_run,
         **kwargs):
    """Evaluates networks using Tensorflow estimators."""
    log_parameters(logger)

    real_name = 'eval_' + name if name else 'eval'
    eval_dir = os.path.join(estimator.model_dir, real_name)
    os.makedirs(eval_dir, exist_ok=True)
    evaluated_file = os.path.join(eval_dir, 'evaluated')
    wait_interval_count = 0
    evaluated_steps_count = 0
    while True:
        evaluated_steps = {}
        ckpt = tf.train.get_checkpoint_state(estimator.model_dir)

        if os.path.exists(evaluated_file):
            evaluated_steps = read_evaluated_file(evaluated_file)

            # create exceptions so we don't delete them
            exceptions = []
            if ckpt and ckpt.model_checkpoint_path:
                for checkpoint_path in ckpt.all_model_checkpoint_paths:
                    try:
                        global_step = str(get_global_step(checkpoint_path))
                    except Exception:
                        logger.warning("Failed to find global_step.", exc_info=True)
                        continue
                    if global_step not in evaluated_steps:
                        exceptions.append(global_step)

            if max_wait_intervals > 0:
                new_evaluated_count = len(evaluated_steps.keys())
                if new_evaluated_count > 0:
                    if new_evaluated_count == evaluated_steps_count:
                        wait_interval_count += 1
                        if wait_interval_count > max_wait_intervals:
                            click.echo("Reached maximum wait interval!")
                            break
                    else:
                        evaluated_steps_count = new_evaluated_count
                        wait_interval_count = 0

            # Save the best N models into the eval directory
            save_n_best_models(estimator.model_dir, eval_dir, evaluated_file,
                               keep_n_best_models, sort_by, exceptions)

        if (not ckpt) or (not ckpt.model_checkpoint_path):
            if max_wait_intervals > 0:
                wait_interval_count += 1
                if wait_interval_count > max_wait_intervals:
                    break
            time.sleep(eval_interval_secs)
            continue

        for checkpoint_path in ckpt.all_model_checkpoint_paths:
            try:
                global_step = str(get_global_step(checkpoint_path))
            except Exception:
                logger.warning(
                    'Failed to find global_step for checkpoint_path {}, '
                    'skipping ...'.format(checkpoint_path), exc_info=True)
                continue
            if global_step in evaluated_steps and not force_re_run:
                continue

            # copy over the checkpoint before evaluating since it might
            # disappear after evaluation.
            try:
                copy_one_step(estimator.model_dir, global_step, eval_dir, fail_on_error=True)
            except Exception:
                # skip testing this checkpoint
                continue

            # evaluate based on the just copied checkpoint_path
            checkpoint_path = checkpoint_path.replace(estimator.model_dir, eval_dir + os.sep)
            checkpoint_path = os.path.abspath(checkpoint_path)
            logger.debug("Evaluating the model from %s", checkpoint_path)

            # Evaluate
            try:
                evaluations = estimator.evaluate(
                    input_fn=eval_input_fn,
                    steps=None,
                    hooks=hooks,
                    checkpoint_path=checkpoint_path,
                    name=name,
                )
            except Exception:
                logger.info("Something went wrong in evaluation.")
                raise

            str_evaluations = append_evaluated_file(evaluated_file,
                                                    evaluations)
            click.echo(str_evaluations)
            sys.stdout.flush()

            # Save the best N models into the eval directory
            save_n_best_models(estimator.model_dir, eval_dir, evaluated_file,
                               keep_n_best_models, sort_by)

        if run_once:
            break
        time.sleep(eval_interval_secs)
