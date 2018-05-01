import os
import shutil
import pkg_resources
import tempfile
from click.testing import CliRunner

from bob.learn.tensorflow.script.db_to_tfrecords import db_to_tfrecords

regenerate_reference = False

dummy_config = pkg_resources.resource_filename(
    'bob.learn.tensorflow', 'test/data/dummy_verify_config.py')


def test_db_to_tfrecords():
    test_dir = tempfile.mkdtemp(prefix='bobtest_')
    output_path = os.path.join(test_dir, 'dev.tfrecords')

    try:
        runner = CliRunner()
        result = runner.invoke(db_to_tfrecords, args=(
            dummy_config, '--output', output_path))
        assert result.exit_code == 0, result.output

        # TODO: test if the generated tfrecords file is equal with a reference
        # file

    finally:
        shutil.rmtree(test_dir)


# def test_db_to_tfrecords_size_estimate():
#     test_dir = tempfile.mkdtemp(prefix='bobtest_')
#     output_path = os.path.join(test_dir, 'dev.tfrecords')
#
#     try:
#         runner = CliRunner()
#         args = (dummy_config, '--size-estimate', '--output', output_path)
#         print(' '.join(args))
#         result = runner.invoke(db_to_tfrecords, args=args,)
#         assert result.exit_code == 0, '%s\n%s\n%s' % (
#             result.exc_info, result.output, result.exception)
#         assert '2.0 M bytes' in result.output, result.output
#
#     finally:
#         shutil.rmtree(test_dir)
