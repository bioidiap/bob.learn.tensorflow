#!/usr/bin/env python

"""Returns predictions of networks trained with
tf.train.MonitoredTrainingSession

Usage:
    %(prog)s [-v...] [-k KEY]... [options] <config_files>...
    %(prog)s --help
    %(prog)s --version

Arguments:
    <config_files>                  The configuration files. The configuration
                                    files are loaded in order and they need to
                                    have several objects inside totally. See
                                    below for explanation.

Options:
    -h --help                       Show this help message and exit
    --version                       Show version and exit
    -o PATH, --output-dir PATH      Name of the output file.
    -k KEY, --predict-keys KEY      List of `str`, name of the keys to predict.
                                    It is used if the
                                    `EstimatorSpec.predictions` is a `dict`. If
                                    `predict_keys` is used then rest of the
                                    predictions will be filtered from the
                                    dictionary. If `None`, returns all.
    --checkpoint-path=<path>        Path of a specific checkpoint to predict.
                                    If `None`, the latest checkpoint in
                                    `model_dir` is used.
    --multiple-samples              If provided, it assumes that the db
                                    interface returns several samples from a
                                    biofile. This option can be used when you
                                    are working with sequences.
    -p N, --number-of-parallel-jobs N  The number of parallel jobs that this
                                       script is run in the SGE grid. You
                                       should use this option with
                                       ``jman submit -t N``.
    -v, --verbose                   Increases the output verbosity level

The configuration files should have the following objects totally:

    # Required objects:

    estimator
    database
    preprocessor
    groups
    biofile_to_label
    bio_predict_input_fn

    # Optional objects:

    hooks

For an example configuration, please see:
bob.learn.tensorflow/bob/learn/tensorflow/examples/mnist/mnist_config.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import os
from multiprocessing import Pool
from collections import defaultdict
import six
import numpy as np
import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.bio.base.utils import read_config_file, save
from bob.bio.base.tools.grid import indices
from bob.learn.tensorflow.dataset import tf_repeat
from bob.learn.tensorflow.utils.commandline import \
    get_from_config_or_commandline
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


def bio_generator(database, preprocessor, groups, number_of_parallel_jobs,
                  biofile_to_label, multiple_samples=False):
    biofiles = list(database.all_files(groups))
    if number_of_parallel_jobs > 1:
        start, end = indices(biofiles, number_of_parallel_jobs)
        biofiles = biofiles[start:end]
    keys = (str(f.make_path("", "")) for f in biofiles)
    labels = (biofile_to_label(f) for f in biofiles)

    def load_data(f, preprocessor, database):
        data = preprocessor.read_original_data(
            f,
            database.original_directory,
            database.original_extension)
        data = preprocessor(data, database.annotations(f))
        return data

    def generator():
        for f, label, key in six.moves.zip(biofiles, labels, keys):
            data = load_data(f, preprocessor, database)
            if multiple_samples:
                label = tf_repeat([label], len(data))
                key = tf_repeat([key], len(data))
            yield (data, label, key)

    # load one data to get its type and shape
    data = load_data(biofiles[0], preprocessor, database)
    data = tf.convert_to_tensor(data)
    output_types = (data.dtype, tf.int64, tf.string)
    data_shape = list(data.shape)
    label_shape = tf.TensorShape([])
    key_shape = tf.TensorShape([])
    if multiple_samples:
        data_shape[0] = None
        label_shape = tf.TensorShape([None])
        key_shape = tf.TensorShape([None])
    output_shapes = (tf.TensorShape(data_shape),
                     label_shape, key_shape)

    return (generator, output_types, output_shapes)


def save_predictions(pool, output_dir, key, pred_buffer):
    outpath = os.path.join(output_dir, key + '.hdf5')
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
    hooks = getattr(config, 'hooks', None)

    # Sets-up logging
    set_verbosity_level(logger, verbosity)

    # required arguments
    estimator = config.estimator
    database = config.database
    preprocessor = config.preprocessor
    groups = config.groups
    biofile_to_label = config.biofile_to_label
    bio_predict_input_fn = config.bio_predict_input_fn
    output_dir = get_from_config_or_commandline(
        config, 'output_dir', args, defaults, False)

    generator, output_types, output_shapes = bio_generator(
        database, preprocessor, groups, number_of_parallel_jobs,
        biofile_to_label, multiple_samples)

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
            key = pred['keys']
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
