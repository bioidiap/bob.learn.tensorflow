#!/usr/bin/env python

"""Trains the VGG-audio network on the AVspoof database.

Usage:
  %(prog)s [options] <checkpoint_dir> <eval_tfrecords>...

Options:
  -h --help                  Show this screen.
  --eval-dir PATH            [default: /idiap/user/amohammadi/avspoof/specgram/avspoof-simple-cnn-eval]
  --input-shape N            [default: (50, 1024, 1)]
  --batch-size N             [default: 50]
  --run-once                 Evaluate the model once only.
  --eval-interval-secs N     Interval to evaluations. [default: 300]
  --num-examples N           Number of examples to run. [default: None] Provide
                             ``None`` to consider all examples.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# for bob imports to work properly:
import pkg_resources
import os
import time
from functools import partial
import tensorflow as tf
from docopt import docopt
from bob.io.base import create_directories_safe
from bob.dap.voice.architectures.simple_cnn import architecture
from bob.dap.base.database.tfrecords import example_parser
from bob.dap.base.util.eval import get_global_step, eval_once


def main(argv=None):
    arguments = docopt(__doc__, argv=argv)
    print(arguments)
    input_shape = eval(arguments['--input-shape'])
    tfrecord_filenames = arguments['<eval_tfrecords>']
    eval_dir = arguments['--eval-dir']
    batch_size = eval(arguments['--batch-size'])
    run_once = arguments['--run-once']
    eval_interval_secs = eval(arguments['--eval-interval-secs'])
    checkpoint_dir = arguments['<checkpoint_dir>']
    num_examples = eval(arguments['--num-examples'])

    create_directories_safe(eval_dir)
    with tf.Graph().as_default() as g:

        # Get images and labels
        with tf.name_scope('input'):
            dataset = tf.contrib.data.TFRecordDataset(tfrecord_filenames)
            feature = {'train/data': tf.FixedLenFeature([], tf.string),
                       'train/label': tf.FixedLenFeature([], tf.int64)}
            my_example_parser = partial(
                example_parser, feature=feature, data_shape=input_shape)
            dataset = dataset.map(
                my_example_parser, num_threads=1, output_buffer_size=batch_size)
            dataset = dataset.batch(batch_size)
            images, labels = dataset.make_one_shot_iterator().get_next()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = architecture(images, mode=tf.estimator.ModeKeys.EVAL)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver()
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir, g)
        evaluated_file = os.path.join(eval_dir, 'evaluated')

        while True:
            evaluated_steps = []
            if os.path.exists(evaluated_file):
                with open(evaluated_file) as f:
                    evaluated_steps = f.read().split()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                for path in ckpt.all_model_checkpoint_paths:
                    global_step = get_global_step(path)
                    if global_step not in evaluated_steps:
                        ret_val = eval_once(saver, summary_writer, top_k_op, summary_op,
                                            path, global_step, num_examples,
                                            batch_size)
                        if ret_val == 0:
                            with open(evaluated_file, 'a') as f:
                                f.write(global_step + '\n')
            if run_once:
                break
            time.sleep(eval_interval_secs)


if __name__ == '__main__':
    main()
