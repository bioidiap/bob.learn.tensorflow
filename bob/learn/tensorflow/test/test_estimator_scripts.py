from __future__ import print_function
import os
from tempfile import mkdtemp
import shutil
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from bob.io.base.test_utils import datafile

from bob.learn.tensorflow.script.db_to_tfrecords import main as tfrecords
from bob.bio.base.script.verify import main as verify
from bob.learn.tensorflow.script.train_generic import main as train_generic
from bob.learn.tensorflow.script.eval_generic import main as eval_generic

dummy_tfrecord_config = datafile('dummy_verify_config.py', __name__)
CONFIG = '''
import tensorflow as tf
from bob.learn.tensorflow.utils.tfrecords import shuffle_data_and_labels, \
    batch_data_and_labels

model_dir = "%(model_dir)s"
tfrecord_filenames = ['%(tfrecord_filenames)s']
data_shape = (1, 112, 92)  # size of atnt images
data_type = tf.uint8
batch_size = 2
epochs = 1
learning_rate = 0.00001
run_once = True


def train_input_fn():
    return shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size, epochs=epochs)

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
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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
'''


def _create_tfrecord(test_dir):
    config_path = os.path.join(test_dir, 'tfrecordconfig.py')
    with open(dummy_tfrecord_config) as f, open(config_path, 'w') as f2:
        f2.write(f.read().replace('TEST_DIR', test_dir))
    verify([config_path])
    tfrecords([config_path])
    return os.path.join(test_dir, 'sub_directory', 'dev.tfrecords')


def _create_checkpoint(tmpdir, model_dir, dummy_tfrecord):
    config = CONFIG % {'model_dir': model_dir,
                       'tfrecord_filenames': dummy_tfrecord}
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    train_generic([config_path])


def _eval(tmpdir, model_dir, dummy_tfrecord):
    config = CONFIG % {'model_dir': model_dir,
                       'tfrecord_filenames': dummy_tfrecord}
    config_path = os.path.join(tmpdir, 'eval_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    eval_generic([config_path])


def test_eval_once():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')
        eval_dir = os.path.join(model_dir, 'eval')

        print('\nCreating a dummy tfrecord')
        dummy_tfrecord = _create_tfrecord(tmpdir)

        print('Training a dummy network')
        _create_checkpoint(tmpdir, model_dir, dummy_tfrecord)

        print('Evaluating a dummy network')
        _eval(tmpdir, model_dir, dummy_tfrecord)

        evaluated_path = os.path.join(eval_dir, 'evaluated')
        assert os.path.exists(evaluated_path), evaluated_path
        with open(evaluated_path) as f:
            doc = f.read()

        assert '1' in doc, doc
        assert '100' in doc, doc
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
