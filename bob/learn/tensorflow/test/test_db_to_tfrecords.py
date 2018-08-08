import os
import shutil
import pkg_resources
import tempfile
from click.testing import CliRunner
import bob.io.base
from bob.learn.tensorflow.script.db_to_tfrecords import db_to_tfrecords, describe_tf_record
from bob.learn.tensorflow.utils import load_mnist, create_mnist_tfrecord
 
regenerate_reference = False

dummy_config = pkg_resources.resource_filename(
    'bob.learn.tensorflow', 'test/data/dummy_verify_config.py')


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
    shape = (3136,) # I'm saving the thing as float
    batch_size = 1000

    try:
        train_data, train_labels, validation_data, validation_labels = load_mnist()
        bob.io.base.create_directories_safe(os.path.dirname(tfrecord_train))
        create_mnist_tfrecord(
            tfrecord_train, train_data, train_labels, n_samples=6000)

        n_samples, n_labels = describe_tf_record(os.path.dirname(tfrecord_train), shape, batch_size)
        
        assert n_samples == 6000
        assert n_labels == 10

    finally:
        shutil.rmtree(os.path.dirname(tfrecord_train))

