import os
import shutil
import tempfile

import numpy as np
import pkg_resources
import tensorflow as tf
from click.testing import CliRunner

from bob.extension.config import load
from bob.extension.scripts.click_helper import assert_click_runner_result
from bob.io.base import create_directories_safe
from bob.learn.tensorflow.dataset.tfrecords import dataset_from_tfrecord
from bob.learn.tensorflow.script.db_to_tfrecords import datasets_to_tfrecords
from bob.learn.tensorflow.utils import create_mnist_tfrecord
from bob.learn.tensorflow.utils import load_mnist

regenerate_reference = False

dummy_config = pkg_resources.resource_filename(
    "bob.learn.tensorflow", "test/data/db_to_tfrecords_config.py"
)


def compare_datasets(ds1, ds2, sess=None):
    if tf.executing_eagerly():
        for values1, values2 in zip(ds1, ds2):
            values1 = tf.nest.flatten(values1)
            values2 = tf.nest.flatten(values2)
            for v1, v2 in zip(values1, values2):
                if not tf.reduce_all(input_tensor=tf.math.equal(v1, v2)):
                    return False
    else:
        ds1 = tf.compat.v1.data.make_one_shot_iterator(ds1).get_next()
        ds2 = tf.compat.v1.data.make_one_shot_iterator(ds2).get_next()
        while True:
            try:
                values1, values2 = sess.run([ds1, ds2])
            except tf.errors.OutOfRangeError:
                break
            values1 = tf.nest.flatten(values1)
            values2 = tf.nest.flatten(values2)
            for v1, v2 in zip(values1, values2):
                v1, v2 = np.asarray(v1), np.asarray(v2)
                if not np.all(v1 == v2):
                    return False
    return True


def test_datasets_to_tfrecords():
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = "./test"
        args = (dummy_config, "--output", output_path)
        result = runner.invoke(datasets_to_tfrecords, args=args, standalone_mode=False)
        assert_click_runner_result(result)
        # read back the tfrecod
        with tf.compat.v1.Session() as sess:
            dataset2 = dataset_from_tfrecord(output_path)
            dataset1 = load(
                [dummy_config], attribute_name="dataset", entry_point_group="bob"
            )
            assert compare_datasets(dataset1, dataset2, sess)
