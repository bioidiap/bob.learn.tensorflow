#!/usr/bin/env python
"""Evaluates networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import click
import logging
import os
import six
import shutil
import sys
import tensorflow as tf
import time
from glob import glob
from collections import defaultdict, OrderedDict
from ..utils.eval import get_global_step
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)
from bob.io.base import create_directories_safe

logger = logging.getLogger(__name__)


def save_n_best_models(train_dir, save_dir, evaluated_file,
                       keep_n_best_models):
    create_directories_safe(save_dir)
    evaluated = read_evaluated_file(evaluated_file)

    def _key(x):
        x = x[1]
        ac = x.get('accuracy') or 0
        lo = x.get('loss') or 0
        return (lo, ac * -1)

    best_models = OrderedDict(sorted(
        evaluated.items(), key=_key)[:keep_n_best_models])

    # delete the old saved models that are not in top N best anymore
    saved_models = defaultdict(list)
    for path in glob('{}/model.ckpt-*'.format(save_dir)):
        global_step = path.split('model.ckpt-')[1].split('.')[0]
        saved_models[global_step].append(path)

    for global_step, paths in saved_models.items():
        if global_step not in best_models:
            for path in paths:
                logger.info("Deleting `%s'", path)
                os.remove(path)

    # copy over the best models if not already there
    for global_step in best_models:
        for path in glob('{}/model.ckpt-{}.*'.format(train_dir, global_step)):
            dst = os.path.join(save_dir, os.path.basename(path))
            if os.path.isfile(dst):
                continue
            logger.info("Copying `%s' over to `%s'", path, dst)
            shutil.copy(path, dst)

    # create a checkpoint file indicating to the best existing model:
    # 1. filter non-existing models first
    def _filter(x):
        return len(glob('{}/model.ckpt-{}.*'.format(save_dir, x[0]))) > 0
    best_models = OrderedDict(filter(_filter, best_models.items()))

    # 2. create the checkpoint file
    with open(os.path.join(save_dir, 'checkpoint'), 'wt') as f:
        for i, global_step in enumerate(best_models):
            if i == 0:
                f.write('model_checkpoint_path: "model.ckpt-{}"\n'.format(
                    global_step))
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
                v = float(v)
                if 'global_step' in k:
                    v = int(v)
                temp[k] = v
            evaluated[global_step] = temp
    return evaluated


def append_evaluated_file(path, evaluations):
    str_evaluations = ', '.join(
        '%s = %s' % (k, v)
        for k, v in sorted(six.iteritems(evaluations)))
    with open(path, 'a') as f:
        f.write('{} {}\n'.format(evaluations['global_step'],
                                 str_evaluations))
    return str_evaluations


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
@click.option('--eval-interval-secs', cls=ResourceOption, type=click.INT,
              default=60, show_default=True)
@click.option('--name', cls=ResourceOption)
@click.option('--keep-n-best-models', '-K', type=click.INT, cls=ResourceOption,
              default=0, show_default=True)
@verbosity_option(cls=ResourceOption)
def eval(estimator, eval_input_fn, hooks, run_once, eval_interval_secs, name,
         keep_n_best_models, **kwargs):
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
    logger.debug('keep_n_best_models: %s', keep_n_best_models)
    logger.debug('kwargs: %s', kwargs)

    real_name = 'eval_' + name if name else 'eval'
    eval_dir = os.path.join(estimator.model_dir, real_name)
    evaluated_file = os.path.join(eval_dir, 'evaluated')
    while True:
        evaluated_steps = {}
        if os.path.exists(evaluated_file):
            evaluated_steps = read_evaluated_file(evaluated_file)

            # Save the best N models into the eval directory
            save_n_best_models(estimator.model_dir, eval_dir, evaluated_file,
                               keep_n_best_models)

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

            str_evaluations = append_evaluated_file(
                evaluated_file, evaluations)
            click.echo(str_evaluations)
            sys.stdout.flush()

            # Save the best N models into the eval directory
            save_n_best_models(estimator.model_dir, eval_dir, evaluated_file,
                               keep_n_best_models)

        if run_once:
            break
        time.sleep(eval_interval_secs)
