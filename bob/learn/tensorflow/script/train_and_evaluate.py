#!/usr/bin/env python
"""Trains and evaluates a network using Tensorflow estimators.
This script calls the estimator.train_and_evaluate function. Please see:
https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec
https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec
for more details.

Usage:
    %(prog)s [-v...] [options] <config_files>...
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
    -v, --verbose                      Increases the output verbosity level

The configuration files should have the following objects totally:

  ## Required objects:

  estimator
  train_spec
  eval_spec

  ## Optional objects:
  exit_ok_exceptions : [Exception]
    A list of exceptions to exit properly if they occur. If nothing is
    provided, the EarlyStopException is handled by default.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import tensorflow as tf
from bob.extension.config import load as read_config_file
from bob.learn.tensorflow.utils.commandline import \
    get_from_config_or_commandline
from bob.learn.tensorflow.utils.hooks import EarlyStopException
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


def main(argv=None):
    from docopt import docopt
    import os
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    defaults = docopt(docs, argv=[""])
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    # optional arguments
    verbosity = get_from_config_or_commandline(config, 'verbose', args,
                                               defaults)

    # Sets-up logging
    set_verbosity_level(logger, verbosity)

    # required objects
    estimator = config.estimator
    train_spec = config.train_spec
    eval_spec = config.eval_spec

    # optional objects
    exit_ok_exceptions = getattr(config, 'exit_ok_exceptions',
                                 (EarlyStopException,))

    # Train and evaluate
    try:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    except exit_ok_exceptions as e:
        logger.exception(e)
        return


if __name__ == '__main__':
    main()
