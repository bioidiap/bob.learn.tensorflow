#!/usr/bin/env python

"""Trains networks using tf.train.MonitoredTrainingSession

Usage:
  %(prog)s [options] <config_files>...
  %(prog)s --help
  %(prog)s --version

Arguments:
  <config_files>  The configuration files. The configuration files are loaded
                  in order and they need to have several objects inside
                  totally. See below for explanation.

Options:
  -h --help  show this help message and exit
  --version  show version and exit

The configuration files should have the following objects totally:

  ## Required objects:

  model_fn
  train_input_fn

  ## Optional objects:

  model_dir
  run_config
  model_params
  hooks
  steps
  max_steps

For an example configuration, please see:
bob.learn.tensorflow/bob/learn/tensorflow/examples/mnist/mnist_config.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import tensorflow as tf
from bob.bio.base.utils import read_config_file


def main(argv=None):
    from docopt import docopt
    import os
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    model_fn = config.model_fn
    train_input_fn = config.train_input_fn

    model_dir = getattr(config, 'model_dir', None)
    run_config = getattr(config, 'run_config', None)
    model_params = getattr(config, 'model_params', None)
    hooks = getattr(config, 'hooks', None)
    steps = getattr(config, 'steps', None)
    max_steps = getattr(config, 'max_steps', None)

    if run_config is None:
        # by default create reproducible nets:
        from bob.learn.tensorflow.utils.reproducible import session_conf
        run_config = tf.estimator.RunConfig()
        run_config.replace(session_config=session_conf)

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                params=model_params, config=run_config)

    # Train
    nn.train(input_fn=train_input_fn, hooks=hooks, steps=steps,
             max_steps=max_steps)


if __name__ == '__main__':
    main()
