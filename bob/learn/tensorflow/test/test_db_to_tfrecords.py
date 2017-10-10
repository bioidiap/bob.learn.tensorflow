import os
import shutil
import pkg_resources
import tempfile

from bob.learn.tensorflow.script.db_to_tfrecords import main as tfrecords
from bob.bio.base.script.verify import main as verify

regenerate_reference = False

dummy_config = pkg_resources.resource_filename(
    'bob.learn.tensorflow', 'test/data/dummy_verify_config.py')


def test_verify_and_tfrecords():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')

  config_path = os.path.join(test_dir, 'config.py')
  with open(dummy_config) as f, open(config_path, 'w') as f2:
    f2.write(f.read().replace('TEST_DIR', test_dir))

  parameters = [config_path]
  try:
    verify(parameters)
    tfrecords(parameters)

    # TODO: test if tfrecords are equal
    # tfrecords_path = os.path.join(test_dir, 'sub_directory', 'dev.tfrecords')
    # if regenerate_reference:
    #   shutil.copy(tfrecords_path, tfrecords_reference)

  finally:
    shutil.rmtree(test_dir)
