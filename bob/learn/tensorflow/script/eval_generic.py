#!/usr/bin/env python

"""Evaluates networks trained with tf.train.MonitoredTrainingSession

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

  model_dir
  model_fn
  eval_input_fn

  ## Optional objects:

  eval_interval_secs
  run_once
  run_config
  model_params
  steps
  hooks
  name

For an example configuration, please see:
bob.learn.tensorflow/bob/learn/tensorflow/examples/mnist/mnist_config.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import os
import time
import six
import tensorflow as tf
from bob.bio.base.utils import read_config_file
from ..utils.eval import get_global_step


def main(argv=None):
    from docopt import docopt
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    model_dir = config.model_dir
    model_fn = config.model_fn
    eval_input_fn = config.eval_input_fn

    eval_interval_secs = getattr(config, 'eval_interval_secs', 300)
    run_once = getattr(config, 'run_once', False)
    run_config = getattr(config, 'run_config', None)
    model_params = getattr(config, 'model_params', None)
    steps = getattr(config, 'steps', None)
    hooks = getattr(config, 'hooks', None)
    name = getattr(config, 'eval_name', None)

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                params=model_params, config=run_config)
    if name:
        real_name = name + '_eval'
    else:
        real_name = 'eval'
    evaluated_file = os.path.join(nn.model_dir, real_name, 'evaluated')
    while True:
        evaluated_steps = []
        if os.path.exists(evaluated_file):
            with open(evaluated_file) as f:
                evaluated_steps = f.read().split()

        ckpt = tf.train.get_checkpoint_state(nn.model_dir)
        if (not ckpt) or (not ckpt.model_checkpoint_path):
            time.sleep(eval_interval_secs)
            continue

        for checkpoint_path in ckpt.all_model_checkpoint_paths:
            global_step = str(get_global_step(checkpoint_path))
            if global_step in evaluated_steps:
                continue

            # Evaluate
            evaluations = nn.evaluate(
                input_fn=eval_input_fn,
                steps=steps,
                hooks=hooks,
                checkpoint_path=checkpoint_path,
                name=name,
            )

            print(', '.join('%s = %s' % (k, v)
                            for k, v in sorted(six.iteritems(evaluations))))
            sys.stdout.flush()
            with open(evaluated_file, 'a') as f:
                f.write('{}\n'.format(evaluations['global_step']))
        if run_once:
            break
        time.sleep(eval_interval_secs)


if __name__ == '__main__':
    main()
