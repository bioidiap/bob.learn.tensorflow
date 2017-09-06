#!/usr/bin/env python

"""Converts Bio and PAD datasets to TFRecords file formats.

Usage:
  %(prog)s <config_files>...
  %(prog)s --help
  %(prog)s --version

Arguments:
  <config_files>  The config files. The config files are loaded in order and
                  they need to have several objects inside totally. See below
                  for explanation.

Options:
  -h --help  show this help message and exit
  --version  show version and exit

The best way to use this script is to send it to the io-big queue if you are at
Idiap:

  $ jman submit -i -q q1d -- python %(prog)s <config_files>...

The config files should have the following objects totally:

  ## Required objects:

  # you need a database object that inherits from
  # bob.bio.base.database.BioDatabase (PAD dbs work too)
  database = Database()

  # the directory pointing to where the preprocessed data is:
  DATA_DIR = '/idiap/temp/user/database_name/sub_directory/preprocessed'

  # the directory to save the tfrecords in:
  OUTPUT_DIR = '/idiap/temp/user/database_name/sub_directory'

  # A function that converts a BioFile or a PadFile to a label:
  # Example for PAD
  def file_to_label(f):
      return f.attack_type is None

  # Example for Bio (You may want to run this script for groups=['world'] only
  # in biometric recognition experiments.)
  CLIENT_IDS = (str(f.client_id) for f in db.all_files(groups=groups))
  CLIENT_IDS = list(set(CLIENT_IDS))
  CLIENT_IDS = dict(zip(CLIENT_IDS, range(1, len(CLIENT_IDS) + 1)))

  def file_to_label(f):
      return CLIENT_IDS[str(f.client_id)]

  ## Optional objects:

  # The groups that you want to create tfrecords for.
  # It should be a list of 'world', 'dev', and 'eval' values.
  groups = ['world']

  # you need a reader function that reads the preprocessed files.
  reader = Preprocessor().read_data
  # or
  from bob.bio.base.utils import load as reader

  # extension of the preprocessed files. Usually '.hdf5'
  DATA_EXTENSION = '.hdf5'

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.bio.base.utils import load, read_config_file
from bob.db.base.utils import check_parameters_for_validity
import bob.core
logger = bob.core.log.setup(__name__)


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main(argv=None):
  from docopt import docopt
  import os
  import sys
  prog = os.path.basename(sys.argv[0])
  args = docopt(__doc__ % {'prog': prog}, argv=argv, version='0.0.1')
  config_files = args['<config_files>']
  config = read_config_file(config_files)

  # Sets-up logging
  verbosity = getattr(config, 'verbose', 0)
  bob.core.log.set_verbosity_level(logger, verbosity)

  database = config.database
  data_dir, output_dir = config.DATA_DIR, config.OUTPUT_DIR
  file_to_label = config.file_to_label

  reader = getattr(config, 'reader', load)
  groups = getattr(config, 'groups', 'world')
  data_extension = getattr(config, 'DATA_EXTENSION', '.hdf5')

  create_directories_safe(output_dir)
  groups = check_parameters_for_validity(
      groups, 'groups', ('world', 'dev', 'eval'), ('world',))

  for group in groups:
    output_file = os.path.join(output_dir, '{}.tfrecords'.format(group))
    files = database.all_files(groups=group)
    n_files = len(files)
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for i, f in enumerate(files):
        logger.info('Processing file %d out of %d', i + 1, n_files)

        path = f.make_path(data_dir, data_extension)
        data = reader(path).astype('float32').tostring()

        feature = {'train/image': _bytes_feature(data),
                   'train/label': _int64_feature(file_to_label(f))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


if __name__ == '__main__':
  main()
