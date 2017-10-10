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

  # checkpoint_dir
  checkpoint_dir = 'train'
  eval_dir = 'eval'
  batch_size = 50
  data, labels = get_data_and_labels()
  logits = architecture(data)

  ## Optional objects:

  num_examples
  run_once
  eval_interval_secs

Example configuration::

    import tensorflow as tf

    checkpoint_dir = 'avspoof-simple-cnn-train'
    eval_dir = 'avspoof-simple-cnn-eval'
    tfrecord_filenames = ['/path/to/dev.tfrecods']
    data_shape = (50, 1024, 1)
    data_type = tf.float32
    batch_size = 50

    from bob.learn.tensorflow.utils.tfrecords import batch_data_and_labels
    def get_data_and_labels():
      return batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size)

    from bob.pad.voice.architectures.simple_cnn import architecture
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import os
import time
import tensorflow as tf
from bob.bio.base.utils import read_config_file
from ..utils.eval import get_global_step, eval_once


def main(argv=None):
    from docopt import docopt
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    run_once = getattr(config, 'run_once', False)
    eval_interval_secs = getattr(config, 'eval_interval_secs', 300)
    num_examples = getattr(config, 'num_examples', None)

    with tf.Graph().as_default() as graph:

        # Get data and labels
        with tf.name_scope('input'):
            data, labels = config.get_data_and_labels()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = config.architecture(data)
        tf.add_to_collection('logits', logits)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(config.eval_dir, graph)
        evaluated_file = os.path.join(config.eval_dir, 'evaluated')

        while True:
            evaluated_steps = []
            if os.path.exists(evaluated_file):
                with open(evaluated_file) as f:
                    evaluated_steps = f.read().split()
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                for path in ckpt.all_model_checkpoint_paths:
                    global_step = get_global_step(path)
                    if global_step not in evaluated_steps:
                        ret_val = eval_once(saver, summary_writer, top_k_op,
                                            path, global_step,
                                            num_examples,
                                            config.batch_size)
                        if ret_val == 0:
                            with open(evaluated_file, 'a') as f:
                                f.write(global_step + '\n')
            if run_once:
                break
            time.sleep(eval_interval_secs)


if __name__ == '__main__':
    main()
