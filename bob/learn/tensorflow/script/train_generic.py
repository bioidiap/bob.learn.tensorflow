#!/usr/bin/env python

"""Trains networks using Tensorflow estimators.

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
    -s N, --steps N                    The number of steps to train.
    -m N, --max-steps N                The maximum number of steps to train.
                                       This is a limit for global step which
                                       continues in separate runs.

The configuration files should have the following objects totally:

  ## Required objects:

  estimator
  train_input_fn

  ## Optional objects:

  hooks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
from bob.bio.base.utils import read_config_file
from bob.learn.tensorflow.utils.commandline import \
    get_from_config_or_commandline
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
    verbosity = get_from_config_or_commandline(
        config, 'verbose', args, defaults)
    max_steps = get_from_config_or_commandline(
        config, 'max_steps', args, defaults)
    steps = get_from_config_or_commandline(
        config, 'steps', args, defaults)
    hooks = getattr(config, 'hooks', None)

    # Sets-up logging
    set_verbosity_level(logger, verbosity)

    # required arguments
    estimator = config.estimator
    train_input_fn = config.train_input_fn

    # Train
    estimator.train(input_fn=train_input_fn, hooks=hooks, steps=steps,
                    max_steps=max_steps)


if __name__ == '__main__':
    main()
