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
    -v, --verbose                   Increases the output verbosity level

The configuration files should have the following objects totally:

    # Required objects:

    estimator
    predict_input_fn

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
import numpy as np
from bob.io.base import create_directories_safe
from bob.bio.base.utils import read_config_file, save
from bob.learn.tensorflow.utils.commandline import \
    get_from_config_or_commandline
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


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
    hooks = getattr(config, 'hooks', None)

    # Sets-up logging
    set_verbosity_level(logger, verbosity)

    # required arguments
    estimator = config.estimator
    predict_input_fn = config.predict_input_fn
    output_dir = get_from_config_or_commandline(
        config, 'output_dir', args, defaults, False)

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
