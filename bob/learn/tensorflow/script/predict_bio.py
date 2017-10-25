#!/usr/bin/env python

"""Saves predictions or embeddings of tf.estimators. This script works with
bob.bio.base databases. To use it see the configuration details below. This
script works with tensorflow 1.4 and above.

Usage:
    %(prog)s [-v...] [-k KEY]... [options] <config_files>...
    %(prog)s --help
    %(prog)s --version

Arguments:
    <config_files>                     The configuration files. The
                                       configuration files are loaded in order
                                       and they need to have several objects
                                       inside totally. See below for
                                       explanation.

Options:
    -h --help                          Show this help message and exit
    --version                          Show version and exit
    -o PATH, --output-dir PATH         Name of the output file.
    -k KEY, --predict-keys KEY         List of `str`, name of the keys to
                                       predict. It is used if the
                                       `EstimatorSpec.predictions` is a `dict`.
                                       If `predict_keys` is used then rest of
                                       the predictions will be filtered from
                                       the dictionary. If `None`, returns all.
    --checkpoint-path=<path>           Path of a specific checkpoint to
                                       predict. If `None`, the latest
                                       checkpoint in `model_dir` is used.
    --multiple-samples                 If provided, it assumes that the db
                                       interface returns several samples from a
                                       biofile. This option can be used when
                                       you are working with sequences.
    -p N, --number-of-parallel-jobs N  The number of parallel jobs that this
                                       script is run in the SGE grid. You
                                       should use this option with ``jman
                                       submit -t N``.
    -f, --force                        If provided, it will overwrite the
                                       existing predictions.
    -v, --verbose                      Increases the output verbosity level

The -- options above can also be supplied through configuration files. You just
need to create a variable with a name that replaces ``-`` with ``_``. For
example, use ``multiple_samples`` instead of ``--multiple-samples``.

The configuration files should have the following objects totally:

    # Required objects:

    estimator : :any:`tf.estimator.Estimator`
        An estimator instance that represents the neural network.
    database : :any:`bob.bio.base.database.BioDatabase`
        A bio database. Its original_directory must point to the correct path.
    groups : [str]
        A list of groups to evaluate. Can be any permutation of
        ``('world', 'dev', 'eval')``.
    bio_predict_input_fn : callable
        A callable with the signature of
        ``input_fn = bio_predict_input_fn(generator, output_types, output_shapes)``
        The inputs are documented in :any:`tf.data.Dataset.from_generator` and
        the output should be a function with no arguments and is passed to
        :any:`tf.estimator.Estimator.predict`.

    # Optional objects:

    read_original_data : callable
        A callable with the signature of
        ``data = read_original_data(biofile, directory, extension)``.
        :any:`bob.bio.base.read_original_data` is used by default.
    hooks : [:any:`tf.train.SessionRunHook`]
        Optional hooks that you may want to attach to the predictions.

An example configuration for a trained model and its evaluation could be::

    import tensorflow as tf

    # define the database:
    from bob.bio.base.test.dummy.database import database

    # load the estimator model
    estimator = tf.estimator.Estimator(model_fn, model_dir)

    groups = ['dev']


    # the ``dataset = tf.data.Dataset.from_generator(generator, output_types,
    # output_shapes)`` line is mandatory in the function below. You have to
    # create it in your configuration file since you want it to be created in
    # the same graph as your model.
    def bio_predict_input_fn(generator,output_types, output_shapes):
        def input_fn():
            dataset = tf.data.Dataset.from_generator(generator, output_types,
                                                     output_shapes)
            # apply all kinds of transformations here, process the data even
            # further if you want.
            dataset = dataset.prefetch(1)
            dataset = dataset.batch(10**3)
            images, labels, keys = dataset.make_one_shot_iterator().get_next()

            return {'data': images, 'keys': keys}, labels
        return input_fn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import os
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from bob.io.base import create_directories_safe
from bob.bio.base.utils import read_config_file, save
from bob.learn.tensorflow.utils.commandline import \
    get_from_config_or_commandline
from bob.learn.tensorflow.dataset.bio import make_output_path, bio_generator
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


def save_predictions(pool, output_dir, key, pred_buffer):
    outpath = make_output_path(output_dir, key)
    create_directories_safe(os.path.dirname(outpath))
    pool.apply_async(save, (np.mean(pred_buffer[key], axis=0), outpath))


def main(argv=None):
    from docopt import docopt
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    defaults = docopt(docs, argv=[""])
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    # optional arguments
    verbosity = get_from_config_or_commandline(
        config, 'verbose', args, defaults)
    predict_keys = get_from_config_or_commandline(
        config, 'predict_keys', args, defaults)
    checkpoint_path = get_from_config_or_commandline(
        config, 'checkpoint_path', args, defaults)
    multiple_samples = get_from_config_or_commandline(
        config, 'multiple_samples', args, defaults)
    number_of_parallel_jobs = get_from_config_or_commandline(
        config, 'number_of_parallel_jobs', args, defaults)
    force = get_from_config_or_commandline(
        config, 'force', args, defaults)
    hooks = getattr(config, 'hooks', None)
    read_original_data = getattr(config, 'read_original_data', None)

    # Sets-up logging
    set_verbosity_level(logger, verbosity)

    # required arguments
    estimator = config.estimator
    database = config.database
    groups = config.groups
    bio_predict_input_fn = config.bio_predict_input_fn
    output_dir = get_from_config_or_commandline(
        config, 'output_dir', args, defaults, False)

    generator, output_types, output_shapes = bio_generator(
        database, groups, number_of_parallel_jobs, output_dir,
        read_original_data=read_original_data, biofile_to_label=None,
        multiple_samples=multiple_samples, force=force)

    predict_input_fn = bio_predict_input_fn(generator,
                                            output_types, output_shapes)

    predictions = estimator.predict(
        predict_input_fn,
        predict_keys=predict_keys,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
    )

    pool = Pool()
    try:
        pred_buffer = defaultdict(list)
        for i, pred in enumerate(predictions):
            key = pred['key']
            prob = pred.get('probabilities', pred.get('embeddings'))
            pred_buffer[key].append(prob)
            if i == 0:
                last_key = key
            if last_key == key:
                continue
            else:
                save_predictions(pool, output_dir, last_key, pred_buffer)
                last_key = key
        # else below is for the for loop
        else:
            save_predictions(pool, output_dir, key, pred_buffer)
    finally:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
