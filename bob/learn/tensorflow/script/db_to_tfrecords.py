#!/usr/bin/env python

"""Converts Bio and PAD datasets to TFRecords file formats.

Usage:
  %(prog)s <config_files>... [--allow-missing-files]
  %(prog)s --help
  %(prog)s --version

Arguments:
  <config_files>  The configuration files. The configuration files are loaded
                  in order and they need to have several objects inside
                  totally. See below for explanation.

Options:
  -h --help  show this help message and exit
  --version  show version and exit

The best way to use this script is to send it to the io-big queue if you are at
Idiap:

  $ jman submit -i -q q1d -- bin/python %(prog)s <config_files>...

The configuration files should have the following objects totally:

  ## Required objects:

  # you need a database object that inherits from
  # bob.bio.base.database.BioDatabase (PAD dbs work too)
  database = Database()

  # the directory pointing to where the processed data is:
  data_dir = '/idiap/temp/user/database_name/sub_directory/preprocessed'

  # the directory to save the tfrecords in:
  output_dir = '/idiap/temp/user/database_name/sub_directory'

  # A function that converts a BioFile or a PadFile to a label:
  # Example for PAD
  def file_to_label(f):
      return f.attack_type is None

  # Example for Bio (You may want to run this script for groups=['world'] only
  # in biometric recognition experiments.)
  CLIENT_IDS = (str(f.client_id) for f in db.all_files(groups=groups))
  CLIENT_IDS = list(set(CLIENT_IDS))
  CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))

  def file_to_label(f):
      return CLIENT_IDS[str(f.client_id)]

  ## Optional objects:

  # The groups that you want to create tfrecords for. It should be a list of
  # 'world' ('train' in bob.pad.base), 'dev', and 'eval' values. [default:
  # 'world']
  groups = ['world']

  # you need a reader function that reads the preprocessed files. [default:
  # bob.bio.base.utils.load]
  reader = Preprocessor().read_data
  reader = Extractor().read_feature
  # or
  from bob.bio.base.utils import load as reader
  # or a reader that casts images to uint8:
  def reader(path):
    data = bob.bio.base.utils.load(path)
    return data.astype("uint8")

  # extension of the preprocessed files. [default: '.hdf5']
  data_extension = '.hdf5'

  # Shuffle the files before writing them into a tfrecords. [default: False]
  shuffle = True

  # Whether the each file contains one sample or more. [default: True] If
  # this is False, the loaded samples from a file are iterated over and each
  # of them is saved as an independent feature.
  one_file_one_sample = True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
# import pkg_resources so that bob imports work properly:
import pkg_resources

import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.bio.base.utils import load, read_config_file
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_a_sample(writer, data, label, feature=None):
    if feature is None:
        feature = {'train/data': bytes_feature(data.tostring()),
                   'train/label': int64_feature(label)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def main(argv=None):
    from docopt import docopt
    import os
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)
    allow_missing_files = args['--allow-missing-files']

    # Sets-up logging
    verbosity = getattr(config, 'verbose', 0)
    set_verbosity_level(logger, verbosity)

    database = config.database
    data_dir, output_dir = config.data_dir, config.output_dir
    file_to_label = config.file_to_label

    reader = getattr(config, 'reader', load)
    groups = getattr(config, 'groups', ['world'])
    data_extension = getattr(config, 'data_extension', '.hdf5')
    shuffle = getattr(config, 'shuffle', False)
    one_file_one_sample = getattr(config, 'one_file_one_sample', True)

    create_directories_safe(output_dir)
    if not isinstance(groups, (list, tuple)):
        groups = [groups]

    for group in groups:
        output_file = os.path.join(output_dir, '{}.tfrecords'.format(group))
        files = database.all_files(groups=group)
        if shuffle:
            random.shuffle(files)
        n_files = len(files)
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for i, f in enumerate(files):
                logger.info('Processing file %d out of %d', i + 1, n_files)

                path = f.make_path(data_dir, data_extension)
                data = reader(path)
                if data is None:
                  if allow_missing_files:
                      logger.debug("... Processing original data file '{0}' was not successful".format(path))
                      continue
                  else:
                      raise RuntimeError("Preprocessing of file '{0}' was not successful".format(path))

                label = file_to_label(f)

                if one_file_one_sample:
                    write_a_sample(writer, data, label)
                else:
                    for sample in data:
                        write_a_sample(writer, sample, label)


if __name__ == '__main__':
    main()
