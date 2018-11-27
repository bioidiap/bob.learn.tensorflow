from __future__ import print_function
import os
import shutil
from glob import glob
from tempfile import mkdtemp
from click.testing import CliRunner
from bob.extension.scripts.click_helper import assert_click_runner_result
from bob.io.base.test_utils import datafile

from bob.learn.tensorflow.script.db_to_tfrecords import db_to_tfrecords
from bob.learn.tensorflow.script.train import train
from bob.learn.tensorflow.script.eval import eval as eval_script
from bob.learn.tensorflow.script.train_and_evaluate import train_and_evaluate
from bob.learn.tensorflow.script.predict_bio import predict_bio


dummy_tfrecord_config = datafile('dummy_verify_config.py', __name__)
CONFIG = '''
import tensorflow as tf
from bob.learn.tensorflow.dataset.tfrecords import shuffle_data_and_labels, \
    batch_data_and_labels

model_dir = "%(model_dir)s"
tfrecord_filenames = ['%(tfrecord_filenames)s']
data_shape = (1, 112, 92)  # size of atnt images
data_type = tf.uint8
batch_size = 2
epochs = 2
learning_rate = 0.00001
run_once = True


def train_input_fn():
    return shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size, epochs=epochs)

def eval_input_fn():
    return batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size, epochs=1)

# config for train_and_evaluate
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=200)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

def architecture(images):
    images = tf.cast(images, tf.float32)
    logits = tf.reshape(images, [-1, 92 * 112])
    logits = tf.layers.dense(inputs=logits, units=20,
                             activation=tf.nn.relu)
    return logits


def model_fn(features, labels, mode, params, config):
    key = features['key']
    features = features['data']

    logits = architecture(features)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "key": key,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    metrics = {'accuracy': accuracy}

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_or_create_global_step())
        # Log accuracy and loss
        with tf.name_scope('train_metrics'):
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('loss', loss)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)


# predict bio config options:

from bob.bio.base.test.dummy.database import database
biofiles = database.all_files(['dev'])

def bio_predict_input_fn(generator, output_types, output_shapes):
    def input_fn():
        dataset = tf.data.Dataset.from_generator(
            generator, output_types, output_shapes)
        # apply all kinds of transformations here, process the data
        # even further if you want.
        dataset = dataset.prefetch(1)
        dataset = dataset.batch(10**3)
        images, labels, keys = dataset.make_one_shot_iterator().get_next()

        return {'data': images, 'key': keys}, labels
    return input_fn
'''


def _create_tfrecord(test_dir):
    config_path = os.path.join(test_dir, 'tfrecordconfig.py')
    with open(dummy_tfrecord_config) as f, open(config_path, 'w') as f2:
        f2.write(f.read().replace('TEST_DIR', test_dir))
    output = os.path.join(test_dir, 'dev.tfrecords')
    runner = CliRunner()
    result = runner.invoke(
        db_to_tfrecords, args=[dummy_tfrecord_config, '--output', output])
    assert_click_runner_result(result)
    return output


def _create_checkpoint(tmpdir, model_dir, dummy_tfrecord):
    config = CONFIG % {
        'model_dir': model_dir,
        'tfrecord_filenames': dummy_tfrecord
    }
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    result = runner.invoke(train, args=[config_path])
    assert_click_runner_result(result)


def _eval(tmpdir, model_dir, dummy_tfrecord, extra_args=[]):
    config = CONFIG % {
        'model_dir': model_dir,
        'tfrecord_filenames': dummy_tfrecord
    }
    config_path = os.path.join(tmpdir, 'eval_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    result = runner.invoke(eval_script, args=[config_path] + extra_args)
    assert_click_runner_result(result)


def _train_and_evaluate(tmpdir, model_dir, dummy_tfrecord):
    config = CONFIG % {
        'model_dir': model_dir,
        'tfrecord_filenames': dummy_tfrecord
    }
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    runner.invoke(train_and_evaluate, args=[config_path])


def _predict_bio(tmpdir, model_dir, dummy_tfrecord, extra_options=tuple()):
    config = CONFIG % {
        'model_dir': model_dir,
        'tfrecord_filenames': dummy_tfrecord
    }
    config_path = os.path.join(tmpdir, 'train_config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    runner = CliRunner()
    return runner.invoke(predict_bio, args=[config_path] + list(extra_options))


def test_eval():
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

        assert '0' in doc, doc
        assert '200' in doc, doc

        print('Train and evaluate a dummy network')
        _train_and_evaluate(tmpdir, model_dir, dummy_tfrecord)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def test_eval_keep_n_model():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')
        eval_dir = os.path.join(model_dir, 'eval')

        print('\nCreating a dummy tfrecord')
        dummy_tfrecord = _create_tfrecord(tmpdir)

        print('Training a dummy network')
        _create_checkpoint(tmpdir, model_dir, dummy_tfrecord)

        print('Evaluating a dummy network')
        _eval(tmpdir, model_dir, dummy_tfrecord, ['-K', '1'])

        evaluated_path = os.path.join(eval_dir, 'evaluated')
        assert os.path.exists(evaluated_path), evaluated_path
        with open(evaluated_path) as f:
            doc = f.read()
        assert '0 ' in doc, doc
        assert '200 ' in doc, doc
        assert len(glob('{}/model.ckpt-*'.format(eval_dir))) == 3, \
            os.listdir(eval_dir)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def test_predict_bio():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')

        dummy_tfrecord = _create_tfrecord(tmpdir)
        _create_checkpoint(tmpdir, model_dir, dummy_tfrecord)

        # Run predict_bio
        result = _predict_bio(
            tmpdir, model_dir, dummy_tfrecord, ['-o', tmpdir, '-vvv'])
        assert_click_runner_result(result)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def test_predict_bio_empty_eval():
    tmpdir = mkdtemp(prefix='bob_')
    try:
        model_dir = os.path.join(tmpdir, 'model_dir')
        eval_dir = os.path.join(model_dir, 'eval')

        dummy_tfrecord = _create_tfrecord(tmpdir)
        _create_checkpoint(tmpdir, model_dir, dummy_tfrecord)

        # Make an empty eval folder
        os.makedirs(eval_dir)
        open(os.path.join(eval_dir, 'checkpoint'), 'w')

        # Run predict_bio
        result = _predict_bio(
            tmpdir, model_dir, dummy_tfrecord,
            ['-o', tmpdir, '-c', eval_dir, '-vvv'])
        # the command should fail when the checkpoint path is empty
        assert_click_runner_result(result, -1)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
