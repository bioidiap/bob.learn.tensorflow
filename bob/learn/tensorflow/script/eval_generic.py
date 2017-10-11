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

  model_fn
  eval_input_fn

  ## Optional objects:

  eval_interval_secs
  run_once
  model_dir
  run_config
  model_params
  steps
  hooks
  name

Example configuration::

    import tensorflow as tf
    from bob.learn.tensorflow.utils.tfrecords import batch_data_and_labels

    model_dir = "%(model_dir)s"
    tfrecord_filenames = ['%(tfrecord_filenames)s']
    data_shape = (1, 112, 92)  # size of atnt images
    data_type = tf.uint8
    batch_size = 2
    epochs = 1
    run_once = True

    def eval_input_fn():
        return batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                     batch_size, epochs=epochs)

    def architecture(images):
        images = tf.cast(images, tf.float32)
        logits = tf.reshape(images, [-1, 92 * 112])
        logits = tf.layers.dense(inputs=logits, units=20,
                                 activation=tf.nn.relu)
        return logits

    def model_fn(features, labels, mode, params, config):
        logits = architecture(features)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by
            # the `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        predictor = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(predictor)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.contrib.framework.get_or_create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                              train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
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

    model_fn = config.model_fn
    eval_input_fn = config.eval_input_fn

    eval_interval_secs = getattr(config, 'eval_interval_secs', 300)
    run_once = getattr(config, 'run_once', False)
    model_dir = getattr(config, 'model_dir', None)
    run_config = getattr(config, 'run_config', None)
    model_params = getattr(config, 'model_params', None)
    steps = getattr(config, 'steps', None)
    hooks = getattr(config, 'hooks', None)
    name = getattr(config, 'eval_name', None)

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                params=model_params, config=run_config)

    evaluated_file = os.path.join(nn.model_dir, name or 'eval', 'evaluated')
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
