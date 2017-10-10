from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
from datetime import datetime


def get_global_step(path):
    """Returns the global number associated with the model checkpoint path. The
    checkpoint must have been saved with the
    :any:`tf.train.MonitoredTrainingSession`.

    Parameters
    ----------
    path : str
        The path to model checkpoint, usually ckpt.model_checkpoint_path

    Returns
    -------
    global_step : str
        The global step number as a string.
    """
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/train/model.ckpt-0,
    # extract global_step from it.
    global_step = path.split('/')[-1].split('-')[-1]
    return global_step


def eval_once(saver, summary_writer, prediction_op, summary_op,
              model_checkpoint_path, global_step, num_examples, batch_size):
    """Run Eval once.

    Parameters
    ----------
    saver
        Saver.
    summary_writer
        Summary writer.
    prediction_op
        Prediction operator.
    summary_op
        Summary operator.
    model_checkpoint_path : str
        Path to the model checkpoint.
    global_step : str
        The global step.
    num_examples : int or None
        The number of examples to try.
    batch_size : int
        The size of evaluation batch.

    This function requires the ``from __future__ import division`` import.

    Returns
    -------
    int
        0 for success, anything else for fail.

    """
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        if model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return -1

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            if num_examples is None:
                num_iter = float("inf")
            else:
                num_iter = int(math.ceil(num_examples / batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([prediction_op])
                true_count += np.sum(predictions)
                total_sample_count += np.asarray(predictions).size
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            return 0
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            return -1
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
