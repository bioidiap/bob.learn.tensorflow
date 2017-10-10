from __future__ import print_function
import os
from tempfile import mkdtemp
import shutil
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from bob.io.base.test_utils import datafile
from bob.io.base import create_directories_safe

from bob.learn.tensorflow.script.db_to_tfrecords import main as tfrecords
from bob.bio.base.script.verify import main as verify
from bob.learn.tensorflow.script.train_generic import main as train_generic
from bob.learn.tensorflow.script.eval_generic import main as eval_generic

dummy_config = datafile('dummy_verify_config.py', __name__)
DATA_SAHPE = (1, 112, 92)


def _create_tfrecord(test_dir):
    config_path = os.path.join(test_dir, 'tfrecordconfig.py')
    with open(dummy_config) as f, open(config_path, 'w') as f2:
        f2.write(f.read().replace('TEST_DIR', test_dir))
    verify([config_path])
    tfrecords([config_path])
    return os.path.join(test_dir, 'sub_directory', 'dev.tfrecords')


def _create_checkpoint(checkpoint_dir, dummy_tfrecord):
    config = '''
import tensorflow as tf

checkpoint_dir = "{}"
tfrecord_filenames = ['{}']
data_shape = {}
data_type = tf.uint8
batch_size = 32
epochs = 1
learning_rate = 0.00001

from bob.learn.tensorflow.utils.tfrecords import shuffle_data_and_labels
def get_data_and_labels():
    return shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size, epochs=epochs)

def architecture(images):
    images = tf.cast(images, tf.float32)
    logits = tf.reshape(images, [-1, 92 * 112])
    logits = tf.layers.dense(inputs=logits, units=20,
                             activation=tf.nn.relu)
    return logits

def loss(logits, labels):
    predictor = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(predictor)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
'''.format(checkpoint_dir, dummy_tfrecord, DATA_SAHPE)
    create_directories_safe(checkpoint_dir)
    config_path = os.path.join(checkpoint_dir, 'config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    train_generic([config_path])


def _eval(checkpoint_dir, eval_dir, dummy_tfrecord):
    config = '''
import tensorflow as tf

checkpoint_dir = '{}'
eval_dir = '{}'
tfrecord_filenames = ['{}']
data_shape = {}
data_type = tf.uint8
batch_size = 2
run_once = True

from bob.learn.tensorflow.utils.tfrecords import batch_data_and_labels
def get_data_and_labels():
  return batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                               batch_size)

def architecture(images):
    images = tf.cast(images, tf.float32)
    logits = tf.reshape(images, [-1, 92 * 112])
    logits = tf.layers.dense(inputs=logits, units=20,
                             activation=tf.nn.relu)
    return logits
'''.format(checkpoint_dir, eval_dir, dummy_tfrecord, DATA_SAHPE)
    create_directories_safe(eval_dir)
    config_path = os.path.join(eval_dir, 'config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    eval_generic([config_path])


def test_eval_once():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        checkpoint_dir = os.path.join(tmpdir, 'checkpoint_dir')
        eval_dir = os.path.join(tmpdir, 'eval_dir')

        print('\nCreating a dummy tfrecord')
        dummy_tfrecord = _create_tfrecord(tmpdir)

        print('Training a dummy network')
        _create_checkpoint(checkpoint_dir, dummy_tfrecord)

        print('Evaluating a dummy network')
        _eval(checkpoint_dir, eval_dir, dummy_tfrecord)

        evaluated_path = os.path.join(eval_dir, 'evaluated')
        assert os.path.exists(evaluated_path)
        with open(evaluated_path) as f:
            doc = f.read()

        assert '1' in doc, doc
        assert '7' in doc, doc
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
