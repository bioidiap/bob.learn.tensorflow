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

  # checkpoint_dir
  checkpoint_dir = 'train'
  batch_size
  data, labels = get_data_and_labels()
  logits = architecture(data)
  loss = loss(logits, labels)
  train_op = optimizer.minimize(loss, global_step=global_step)

  ## Optional objects:

  log_frequency
  max_to_keep

Example configuration::

    import tensorflow as tf

    checkpoint_dir = 'avspoof-simple-cnn-train'
    tfrecord_filenames = ['/path/to/group.tfrecod']
    data_shape = (50, 1024, 1)
    data_type = tf.float32
    batch_size = 32
    epochs = None
    learning_rate = 0.00001

    from bob.learn.tensorflow.utils.tfrecods import shuffle_data_and_labels
    def get_data_and_labels():
        return shuffle_data_and_labels(tfrecord_filenames, data_shape,
                                       data_type, batch_size, epochs=epochs)

    from bob.pad.voice.architectures.simple_cnn import architecture

    def loss(logits, labels):
        predictor = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        return tf.reduce_mean(predictor)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
# for creating reproducible nets
from ..utils.reproducible import session_conf

import tensorflow as tf
from bob.bio.base.utils import read_config_file
from ..utils.hooks import LoggerHook


def main(argv=None):
    from docopt import docopt
    import os
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    max_to_keep = getattr(config, 'max_to_keep', 10**5)
    log_frequency = getattr(config, 'log_frequency', 100)

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get data and labels
        with tf.name_scope('input'):
            data, labels = config.get_data_and_labels()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = config.architecture(data)
        tf.add_to_collection('logits', logits)

        # Calculate loss.
        loss = config.loss(logits=logits, labels=labels)
        tf.summary.scalar('loss', loss)

        # get training operation using optimizer:
        train_op = config.optimizer.minimize(loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=max_to_keep)
        scaffold = tf.train.Scaffold(saver=saver)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=config.checkpoint_dir,
                scaffold=scaffold,
                hooks=[
                    tf.train.CheckpointSaverHook(config.checkpoint_dir,
                                                 save_secs=60 * 29,
                                                 scaffold=scaffold),
                    tf.train.NanTensorHook(loss),
                    LoggerHook(loss, config.batch_size, log_frequency)],
                config=session_conf,
                save_checkpoint_secs=None,
                save_summaries_steps=100,
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == '__main__':
    main()
