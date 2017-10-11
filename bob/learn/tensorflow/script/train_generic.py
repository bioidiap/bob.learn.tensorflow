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

Example configuration::

    import tensorflow as tf
    from bob.learn.tensorflow.utils.tfrecords import shuffle_data_and_labels

    model_dir = "%(model_dir)s"
    tfrecord_filenames = ['%(tfrecord_filenames)s']
    data_shape = (1, 112, 92)  # size of atnt images
    data_type = tf.uint8
    batch_size = 2
    epochs = 1
    learning_rate = 0.00001

    def train_input_fn():
        return shuffle_data_and_labels(tfrecord_filenames, data_shape,
                                       data_type, batch_size, epochs=epochs)

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
