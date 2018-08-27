#!/usr/bin/env python
"""Deletes extra tensorflow checkpoints.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import click
import logging
import os
import glob
import tensorflow as tf
from bob.extension.scripts.click_helper import verbosity_option, log_parameters

logger = logging.getLogger(__name__)


def delete_extra_checkpoints(directory, keep_last_n, dry_run):
    try:
        ckpt = tf.train.get_checkpoint_state(directory)
    except Exception:
        return
    if (not ckpt) or (not ckpt.model_checkpoint_path):
        logger.debug('Could not find a checkpoint in %s', directory)
        return
    for checkpoint_path in ckpt.all_model_checkpoint_paths[:-keep_last_n]:
        if checkpoint_path == ckpt.model_checkpoint_path:
            continue
        if dry_run:
            click.echo('Would delete {}.*'.format(checkpoint_path))
        else:
            logger.info('Deleting %s.*', checkpoint_path)
            for path in glob.glob('{}.*'.format(checkpoint_path)):
                os.remove(path)

    def _existing(x):
        return glob.glob('{}.*'.format(x))

    # update the checkpoint file
    all_paths = filter(_existing, ckpt.all_model_checkpoint_paths)
    all_paths = list(map(os.path.basename, all_paths))
    model_checkpoint_path = os.path.basename(ckpt.model_checkpoint_path)
    tf.train.update_checkpoint_state(
        directory, model_checkpoint_path, all_paths)


@click.command(epilog='''\b
Examples:
$ bob tf trim -vv ~/my_models/model_dir
$ bob tf trim -vv ~/my_models/model_dir1 ~/my_models/model_dir2
$ bob tf trim -vvr ~/my_models
$ bob tf trim -vvrn ~/my_models
$ bob tf trim -vvrK 2 ~/my_models
''')
@click.argument(
    'root_dirs',
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    '--keep-last-n-models',
    '-K',
    type=click.INT,
    default=1,
    show_default=True,
    help='The number of recent checkpoints to keep.')
@click.option(
    '--recurse',
    '-r',
    is_flag=True,
    help='If given, it will delete checkpoints recursively.')
@click.option(
    '--dry-run',
    '-n',
    is_flag=True,
    help='If given, will only print what will be deleted.')
@verbosity_option()
def trim(root_dirs, keep_last_n_models, recurse, dry_run, **kwargs):
    """Deletes extra tensorflow checkpoints."""
    log_parameters(logger)

    for root_dir in root_dirs:
        if recurse:
            for directory, _, _ in os.walk(root_dir):
                delete_extra_checkpoints(directory, keep_last_n_models,
                                         dry_run)
        else:
            delete_extra_checkpoints(root_dir, keep_last_n_models, dry_run)
