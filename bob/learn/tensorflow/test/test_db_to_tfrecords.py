import os
import shutil
import pkg_resources
import tempfile
import tensorflow as tf
import numpy as np
from click.testing import CliRunner
from bob.io.base import create_directories_safe
from bob.learn.tensorflow.script.db_to_tfrecords import (
    db_to_tfrecords, describe_tf_record, datasets_to_tfrecords)
from bob.learn.tensorflow.utils import load_mnist, create_mnist_tfrecord
from bob.extension.scripts.click_helper import assert_click_runner_result
from bob.extension.config import load
from bob.learn.tensorflow.dataset.tfrecords import dataset_from_tfrecord

regenerate_reference = False

dummy_config = pkg_resources.resource_filename(
    'bob.learn.tensorflow', 'test/data/db_to_tfrecords_config.py')


def compare_datasets(ds1, ds2, sess=None):
    if tf.executing_eagerly():
        for values1, values2 in zip(ds1, ds2):
            values1 = tf.contrib.framework.nest.flatten(values1)
            values2 = tf.contrib.framework.nest.flatten(values2)
            for v1, v2 in zip(values1, values2):
                if not tf.reduce_all(tf.math.equal(v1, v2)):
                    return False
    else:
        ds1 = ds1.make_one_shot_iterator().get_next()
        ds2 = ds2.make_one_shot_iterator().get_next()
        while True:
            try:
                values1, values2 = sess.run([ds1, ds2])
            except tf.errors.OutOfRangeError:
                break
            values1 = tf.contrib.framework.nest.flatten(values1)
            values2 = tf.contrib.framework.nest.flatten(values2)
            for v1, v2 in zip(values1, values2):
                v1, v2 = np.asarray(v1), np.asarray(v2)
                if not np.all(v1 == v2):
                    return False
    return True


def test_db_to_tfrecords():
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    output_path = os.path.join(test_dir, 'dev.tfrecords')

    try:
        runner = CliRunner()
        result = runner.invoke(
            db_to_tfrecords,
            args=(dummy_config, '--output', output_path),
            standalone_mode=False)
        assert result.exit_code == 0, '%s\n%s\n%s' % (
            result.exc_info, result.output, result.exception)

        # TODO: test if the generated tfrecords file is equal with a reference
        # file

    finally:
        shutil.rmtree(test_dir)


def test_db_to_tfrecords_size_estimate():
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    output_path = os.path.join(test_dir, 'dev.tfrecords')

    try:
        args = (dummy_config, '--size-estimate', '--output', output_path)
        runner = CliRunner()
        result = runner.invoke(
            db_to_tfrecords, args=args, standalone_mode=False)
        assert result.exit_code == 0, '%s\n%s\n%s' % (
            result.exc_info, result.output, result.exception)
        assert '2.0 M bytes' in result.output, result.output

    finally:
        shutil.rmtree(test_dir)


def test_tfrecord_counter():
    tfrecord_train = "./tf-train-test/train_mnist.tfrecord"
    shape = (3136,)  # I'm saving the thing as float
    batch_size = 1000

    try:
        train_data, train_labels, validation_data, validation_labels = \
            load_mnist()
        create_directories_safe(os.path.dirname(tfrecord_train))
        create_mnist_tfrecord(
            tfrecord_train, train_data, train_labels, n_samples=6000)

        n_samples, n_labels = describe_tf_record(
            os.path.dirname(tfrecord_train), shape, batch_size)

        assert n_samples == 6000
        assert n_labels == 10

    finally:
        shutil.rmtree(os.path.dirname(tfrecord_train))


def test_datasets_to_tfrecords():
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = './test'
        args = (dummy_config, '--output', output_path)
        result = runner.invoke(
            datasets_to_tfrecords, args=args, standalone_mode=False)
        assert_click_runner_result(result)
        # read back the tfrecod
        with tf.Session() as sess:
            dataset2 = dataset_from_tfrecord(output_path)
            dataset1 = load(
                [dummy_config], attribute_name='dataset', entry_point_group='bob')
            assert compare_datasets(dataset1, dataset2, sess)
