#!/usr/bin/env python
"""Saves predictions or embeddings of tf.estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import click
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from bob.io.base import create_directories_safe
from bob.bio.base.utils import save
from bob.bio.base.tools.grid import indices
from bob.learn.tensorflow.dataset.bio import BioGenerator

logger = logging.getLogger(__name__)


def make_output_path(output_dir, key):
    """Returns an output path used for saving keys. You need to make sure the
    directories leading to this output path exist.

    Parameters
    ----------
    output_dir : str
        The root directory to save the results
    key : str
        The key of the sample. Usually biofile.make_path("", "")

    Returns
    -------
    str
        The path for the provided key.
    """
    return os.path.join(output_dir, key + '.hdf5')


def non_existing_files(paths, force=False):
    if force:
        for i in range(len(paths)):
            yield i
        return
    for i, path in enumerate(paths):
        if not os.path.isfile(path):
            yield i


def save_predictions(pool, output_dir, key, pred_buffer):
    outpath = make_output_path(output_dir, key)
    create_directories_safe(os.path.dirname(outpath))
    logger.debug("Saving predictions for %s", key)
    pool.apply_async(save, (np.mean(pred_buffer[key], axis=0), outpath))


@click.command(entry_point_group='bob.learn.tensorflow.config',
               cls=ConfigCommand)
@click.option('--estimator', '-e', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.estimator')
@click.option('--database', '-d', required=True, cls=ResourceOption,
              entry_point_group='bob.bio.database')
@click.option('--biofiles', required=True, cls=ResourceOption,
              help='You can only provide this through config files.')
@click.option('--bio-predict-input-fn', required=True, cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.biogenerator_input')
@click.option('--output-dir', '-o', required=True, cls=ResourceOption)
@click.option('--load-data', cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.load_data')
@click.option('--hooks', cls=ResourceOption, multiple=True,
              entry_point_group='bob.learn.tensorflow.hook')
@click.option('--predict-keys', '-k', multiple=True, default=None,
              cls=ResourceOption)
@click.option('--checkpoint-path', cls=ResourceOption)
@click.option('--multiple-samples', is_flag=True, cls=ResourceOption)
@click.option('--array', '-t', type=click.INT, default=1, cls=ResourceOption)
@click.option('--force', '-f', is_flag=True, cls=ResourceOption)
@verbosity_option(cls=ResourceOption)
def predict_bio(estimator, database, biofiles, bio_predict_input_fn,
                output_dir, load_data, hooks, predict_keys, checkpoint_path,
                multiple_samples, array, force, **kwargs):
    """Saves predictions or embeddings of tf.estimators.

    This script works with bob.bio.base databases. This script works with
    tensorflow 1.4 and above.

    \b
    Parameters
    ----------
    estimator : tf.estimator.Estimator
        The estimator that will be trained. Can be a
        ``bob.learn.tensorflow.estimator`` entry point or a path to a Python
        file which contains a variable named `estimator`.
    database : :any:`bob.bio.base.database.BioDatabase`
        A bio database. Its original_directory must point to the correct path.
    biofiles : [:any:`bob.bio.base.database.BioFile`]
        The list of the bio files.
    bio_predict_input_fn : callable
        A callable with the signature of
        ``input_fn = bio_predict_input_fn(generator, output_types, output_shapes)``
        The inputs are documented in :any:`tf.data.Dataset.from_generator` and
        the output should be a function with no arguments and is passed to
        :any:`tf.estimator.Estimator.predict`.
    output_dir : str
        The directory to save the predictions.
    load_data : callable, optional
        A callable with the signature of
        ``data = load_data(database, biofile)``.
        :any:`bob.bio.base.read_original_data` is used by default.
    hooks : [tf.train.SessionRunHook], optional
        List of SessionRunHook subclass instances. Used for callbacks inside
        the training loop. Can be a ``bob.learn.tensorflow.hook`` entry point
        or a path to a Python file which contains a variable named `hooks`.
    predict_keys : [str] or None, optional
        List of `str`, name of the keys to predict. It is used if the
        `EstimatorSpec.predictions` is a `dict`. If `predict_keys` is used then
        rest of the predictions will be filtered from the dictionary. If
        `None`, returns all.
    checkpoint_path : str, optional
        Path of a specific checkpoint to predict. If `None`, the latest
        checkpoint in `model_dir` is used.
    multiple_samples : bool, optional
        If provided, it assumes that the db interface returns several samples
        from a biofile. This option can be used when you are working with
        sequences.
    force : bool, optional
        Whether to overwrite existing predictions.
    array : int, optional
        Use this option alongside gridtk to submit this script as an array job.
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

    An example configuration for a trained model and its evaluation could be::

        import tensorflow as tf

        # define the database:
        from bob.bio.base.test.dummy.database import database

        # load the estimator model
        estimator = tf.estimator.Estimator(model_fn, model_dir)

        groups = ['dev']
        biofiles = database.all_files(groups)


        # the ``dataset = tf.data.Dataset.from_generator(generator,
        # output_types, output_shapes)`` line is mandatory in the function
        # below. You have to create it in your configuration file since you
        # want it to be created in the same graph as your model.
        def bio_predict_input_fn(generator, output_types, output_shapes):
            def input_fn():
                dataset = tf.data.Dataset.from_generator(
                    generator, output_types, output_shapes)
                # apply all kinds of transformations here, process the data
                # even further if you want.
                dataset = dataset.prefetch(1)
                dataset = dataset.batch(10**3)
                images, labels, keys = dataset.make_one_shot_iterator().get_next()

                return {'data': images, 'keys': keys}, labels
            return input_fn
    """
    logger.debug('estimator: %s', estimator)
    logger.debug('database: %s', database)
    logger.debug('len(biofiles): %s', len(biofiles))
    logger.debug('bio_predict_input_fn: %s', bio_predict_input_fn)
    logger.debug('output_dir: %s', output_dir)
    logger.debug('load_data: %s', load_data)
    logger.debug('hooks: %s', hooks)
    logger.debug('predict_keys: %s', predict_keys)
    logger.debug('checkpoint_path: %s', checkpoint_path)
    logger.debug('multiple_samples: %s', multiple_samples)
    logger.debug('array: %s', array)
    logger.debug('force: %s', force)
    logger.debug('kwargs: %s', kwargs)

    assert len(biofiles), "biofiles are empty!"

    if array is not None:
        logger.info("array: %d", array)
    if array > 1:
        start, end = indices(biofiles, array)
        biofiles = biofiles[start:end]

    # filter the existing files
    paths = [make_output_path(output_dir, f.make_path("", ""))
             for f in biofiles]
    indexes = non_existing_files(paths, force)
    biofiles = [biofiles[i] for i in indexes]

    if len(biofiles) == 0:
        logger.warning(
            "The biofiles are empty after checking for existing files.")
        return

    generator = BioGenerator(
        database,
        biofiles,
        load_data=load_data,
        multiple_samples=multiple_samples)

    predict_input_fn = bio_predict_input_fn(generator, generator.output_types,
                                            generator.output_shapes)

    if checkpoint_path:
        logger.info("Restoring the model from %s", checkpoint_path)

    predictions = estimator.predict(
        predict_input_fn,
        predict_keys=predict_keys,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
    )

    logger.info("Saving the predictions of %d files in %s", len(generator),
                output_dir)

    pool = Pool()
    try:
        pred_buffer = defaultdict(list)
        for i, pred in enumerate(predictions):
            key = pred['key']
            # key is in bytes format in Python 3
            if sys.version_info >= (3, ):
                key = key.decode(errors='replace')
            prob = pred.get('probabilities', pred.get('embeddings'))
            pred_buffer[key].append(prob)
            if i == 0:
                last_key = key
            if last_key == key:
                continue
            else:
                save_predictions(pool, output_dir, last_key, pred_buffer)
                last_key = key
        # save the final returned key as well:
        save_predictions(pool, output_dir, key, pred_buffer)
    finally:
        pool.close()
        pool.join()
