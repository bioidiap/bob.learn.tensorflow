#!/usr/bin/env python

"""Trains the VGG-audio network on the AVspoof database.

Usage:
  %(prog)s [options] <train_tfrecords>...

Options:
  -h --help                  Show this screen.
  --save-dir PATH            [default: /idiap/user/amohammadi/avspoof/specgram/avspoof-simple-cnn-train]
  --input-shape N            [default: (50, 1024, 1)]
  --epochs N                 [default: None]
  --batch-size N             [default: 32]
  --capacity-samples N       The capacity of the queue [default: 10**4/2].
  --learning-rate N          The learning rate [default: 0.00001].
  --log-frequency N          How often to log results to the console.
                             [default: 100]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# for bob imports to work properly:
import pkg_resources
# for creating reproducible nets
from bob.dap.base.sandbox.reproducible import session_conf

import tensorflow as tf
from docopt import docopt
from bob.io.base import create_directories_safe
from bob.dap.voice.architectures.simple_cnn import architecture
from bob.dap.base.database.tfrecords import read_and_decode
from bob.dap.base.util.hooks import LoggerHook


def main(argv=None):
    arguments = docopt(__doc__, argv=argv)
    print(arguments)
    input_shape = eval(arguments['--input-shape'])
    tfrecord_filenames = arguments['<train_tfrecords>']
    save_dir = arguments['--save-dir']
    epochs = eval(arguments['--epochs'])
    batch_size = eval(arguments['--batch-size'])
    capacity_samples = eval(arguments['--capacity-samples'])
    learning_rate = eval(arguments['--learning-rate'])
    log_frequency = eval(arguments['--log-frequency'])

    create_directories_safe(save_dir)
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                tfrecord_filenames, num_epochs=epochs, name="tfrecord_filenames")

            image, label = read_and_decode(filename_queue, input_shape)
            images, labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size,
                capacity=capacity_samples // batch_size,
                min_after_dequeue=int(capacity_samples // batch_size // 2),
                num_threads=1, name="shuffle_batch")

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = architecture(images)
        tf.add_to_collection('logits', logits)

        # Calculate loss.
        predictor = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(predictor)
        tf.summary.scalar('loss', loss)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=10**5)
        scaffold = tf.train.Scaffold(saver=saver)
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=save_dir,
                scaffold=scaffold,
                hooks=[
                    tf.train.CheckpointSaverHook(
                        save_dir, save_secs=60 * 29, scaffold=scaffold),
                    tf.train.NanTensorHook(loss),
                    LoggerHook(loss, batch_size, log_frequency)],
                config=session_conf,
                save_checkpoint_secs=None,
                save_summaries_steps=100,
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == '__main__':
    main()
