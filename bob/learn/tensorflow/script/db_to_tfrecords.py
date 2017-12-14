#!/usr/bin/env python
"""Converts Bio and PAD datasets to TFRecords file formats.

Usage:
    %(prog)s [-v...] [options] <config_files>...
    %(prog)s --help
    %(prog)s --version

Arguments:
    <config_files>              The configuration files. The configuration
                                files are loaded in order and they need to have
                                several objects inside totally. See below for
                                explanation.

Options:
    -h --help                   Show this help message and exit
    --version                   Show version and exit
    -o PATH, --output PATH      Name of the output file.
    --shuffle                   If provided, it will shuffle the samples.
    --allow-failures            If provided, the samples which fail to load are
                                ignored.
    --multiple-samples          If provided, it means that the data provided by
                                reader contains multiple samples with same
                                label and path.
    -v, --verbose               Increases the output verbosity level

The best way to use this script is to send it to the io-big queue if you are at
Idiap:

    $ jman submit -i -q q1d -- %(prog)s <config_files>...

The configuration files should have the following objects totally::

    # Required objects:
    samples : a list of all samples that you want to write in the tfrecords
              file. Whatever is inside this list is passed to the reader.
    reader  : a function with the signature of
              ``data, label, key = reader(sample)`` which takes a sample and
              returns the loaded data, the label of the data, and a key which
              is unique for every sample.

You can also provide the command line options in the configuration file too.
It is needed to replace "-" with "_" when they are in the configuration file.
The ones provided by command line overwrite the values of the config file.

An example for mnist would be::

    from bob.db.mnist import Database
    db = Database()
    data, labels = db.data(groups='train')

    samples = zip(data, labels, (str(i) for i in range(len(data))))

    def reader(sample):
        return sample

    allow_failures = True
    output = '/tmp/mnist_train.tfrecords'
    shuffle = True

An example for bob.bio.base would be::

    from bob.bio.base.test.dummy.database import database
    from bob.bio.base.utils import read_original_data

    groups = 'dev'

    samples = database.all_files(groups=groups)

    CLIENT_IDS = (str(f.client_id) for f in database.all_files(groups=groups))
    CLIENT_IDS = list(set(CLIENT_IDS))
    CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))


    def file_to_label(f):
        return CLIENT_IDS[str(f.client_id)]


    def reader(biofile):
        data = read_original_data(
            biofile, database.original_directory, database.original_extension)
        label = file_to_label(biofile)
        key = biofile.path
        return (data, label, key)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
# import pkg_resources so that bob imports work properly:
import pkg_resources
import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.extension.config import load as read_config_file
from bob.learn.tensorflow.utils.commandline import \
    get_from_config_or_commandline
from bob.core.log import setup, set_verbosity_level
logger = setup(__name__)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_a_sample(writer, data, label, key, feature=None):
    if feature is None:
        feature = {
            'data': bytes_feature(data.tostring()),
            'label': int64_feature(label),
            'key': bytes_feature(key)
        }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def main(argv=None):
    from docopt import docopt
    import os
    import sys
    docs = __doc__ % {'prog': os.path.basename(sys.argv[0])}
    version = pkg_resources.require('bob.learn.tensorflow')[0].version
    defaults = docopt(docs, argv=[""])
    args = docopt(docs, argv=argv, version=version)
    config_files = args['<config_files>']
    config = read_config_file(config_files)

    # optional arguments
    verbosity = get_from_config_or_commandline(config, 'verbose', args,
                                               defaults)
    allow_failures = get_from_config_or_commandline(config, 'allow_failures',
                                                    args, defaults)
    multiple_samples = get_from_config_or_commandline(
        config, 'multiple_samples', args, defaults)
    shuffle = get_from_config_or_commandline(config, 'shuffle', args, defaults)

    # Sets-up logging
    set_verbosity_level(logger, verbosity)

    # required arguments
    samples = config.samples
    reader = config.reader
    output = get_from_config_or_commandline(config, 'output', args, defaults,
                                            False)

    if not output.endswith(".tfrecords"):
        output += ".tfrecords"

    create_directories_safe(os.path.dirname(output))

    n_samples = len(samples)
    sample_counter = 0
    with tf.python_io.TFRecordWriter(output) as writer:
        if shuffle:
            random.shuffle(samples)
        for i, sample in enumerate(samples):
            logger.info('Processing file %d out of %d', i + 1, n_samples)

            data, label, key = reader(sample)

            if data is None:
                if allow_failures:
                    logger.debug("... Skipping `{0}`.".format(sample))
                    continue
                else:
                    raise RuntimeError(
                        "Reading failed for `{0}`".format(sample))

            if multiple_samples:
                for sample in data:
                    write_a_sample(writer, sample, label, key)
                    sample_counter += 1
            else:
                write_a_sample(writer, data, label, key)
                sample_counter += 1

    print("Wrote {} samples into the tfrecords file.".format(sample_counter))


if __name__ == '__main__':
    main()
