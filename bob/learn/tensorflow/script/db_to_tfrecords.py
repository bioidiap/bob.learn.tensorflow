#!/usr/bin/env python
"""Converts Bio and PAD datasets to TFRecords file formats.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tempfile
import os
import sys
import logging
import click
import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)
import numpy
from bob.learn.tensorflow.dataset.tfrecords import describe_tf_record


logger = logging.getLogger(__name__)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_a_sample(writer, data, label, key, feature=None,
                   size_estimate=False):
    if feature is None:
        feature = {
            'data': bytes_feature(data.tostring()),
            'label': int64_feature(label),
            'key': bytes_feature(key)
        }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example.SerializeToString()
    if not size_estimate:
        writer.write(example)
    return sys.getsizeof(example)


def _bytes2human(n, format='%(value).1f %(symbol)s', symbols='customary'):
    """Convert n bytes into a human readable string based on format.
    From: https://code.activestate.com/recipes/578019-bytes-to-human-human-to-
    bytes-converter/
    Author: Giampaolo Rodola' <g.rodola [AT] gmail [DOT] com>
    License: MIT
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """
    SYMBOLS = {
        'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
        'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta',
                          'exa', 'zetta', 'iotta'),
        'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
        'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi',
                    'zebi', 'yobi'),
    }
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--samples',
    required=True,
    cls=ResourceOption,
    help='A list of all samples that you want to write in the '
    'tfrecords file. Whatever is inside this list is passed to '
    'the reader.')
@click.option(
    '--reader',
    required=True,
    cls=ResourceOption,
    help='a function with the signature of ``data, label, key = '
    'reader(sample)`` which takes a sample and returns the '
    'loaded data, the label of the data, and a key which is '
    'unique for every sample.')
@click.option(
    '--output',
    '-o',
    required=True,
    cls=ResourceOption,
    help='Name of the output file.')
@click.option(
    '--shuffle',
    is_flag=True,
    cls=ResourceOption,
    help='If provided, it will shuffle the samples.')
@click.option(
    '--allow-failures',
    is_flag=True,
    cls=ResourceOption,
    help='If provided, the samples which fail to load are ignored.')
@click.option(
    '--multiple-samples',
    is_flag=True,
    cls=ResourceOption,
    help='If provided, it means that the data provided by reader contains '
    'multiple samples with same label and path.')
@click.option(
    '--size-estimate',
    is_flag=True,
    cls=ResourceOption,
    help='If given, will print the estimated file size instead of creating '
    'the final tfrecord file.')
@verbosity_option(cls=ResourceOption)
def db_to_tfrecords(samples, reader, output, shuffle, allow_failures,
                    multiple_samples, size_estimate, **kwargs):
    """Converts Bio and PAD datasets to TFRecords file formats.

    The best way to use this script is to send it to the io-big queue if you
    are at Idiap::

        $ jman submit -i -q q1d -- %(prog)s <config_files>...

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

        CLIENT_IDS = (str(f.client_id) for f
                      in database.all_files(groups=groups))
        CLIENT_IDS = list(set(CLIENT_IDS))
        CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))


        def file_to_label(f):
            return CLIENT_IDS[str(f.client_id)]


        def reader(biofile):
            data = read_original_data(
                biofile, database.original_directory,
                database.original_extension)
            label = file_to_label(biofile)
            key = biofile.path
            return (data, label, key)
    """
    log_parameters(logger, ignore=('samples', ))
    logger.debug("len(samples): %d", len(samples))

    if size_estimate:
        output = tempfile.NamedTemporaryFile(suffix='.tfrecords').name

    if not output.endswith(".tfrecords"):
        output += ".tfrecords"

    if not size_estimate:
        logger.info("Writing samples to `{}'".format(output))

    total_size = 0

    create_directories_safe(os.path.dirname(output))

    n_samples = len(samples)
    sample_count = 0
    with tf.python_io.TFRecordWriter(output) as writer:
        if shuffle:
            logger.info("Shuffling the samples before writing ...")
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
                    total_size += write_a_sample(
                        writer,
                        sample,
                        label,
                        key,
                        size_estimate=size_estimate)
                    sample_count += 1
            else:
                total_size += write_a_sample(
                    writer, data, label, key, size_estimate=size_estimate)
                sample_count += 1

    if not size_estimate:
        click.echo(
            "Wrote {} samples into the tfrecords file.".format(sample_count))
    else:
        # delete the empty tfrecords file
        try:
            os.remove(output)
        except Exception:
            pass
    click.echo("The total size of the tfrecords file will be roughly "
               "{} bytes".format(_bytes2human(total_size)))


@click.command()
@click.argument(
    'tf-record-path',
    nargs=1)
@click.argument(
    'shape',
    type=int,
    nargs=-1
)
@click.option(
    '--batch-size',
    help='Batch size',
    show_default=True,
    required=True,
    default=1000
)
@verbosity_option(cls=ResourceOption)
def describe_tfrecord(tf_record_path, shape, batch_size, **kwargs):
    '''
    Very often you have a tf-record file, or a set of them, and you have no
    idea how many samples you have there. Even worse, you have no idea how many
    classes you have.

    This click command will solve this thing for you by doing the following::

        $ %(prog)s <tf-record-path> 182 182 3

    '''
    n_samples, n_labels = describe_tf_record(tf_record_path, shape, batch_size)
    click.echo("#############################################")
    click.echo("Number of samples {0}".format(n_samples))
    click.echo("Number of labels {0}".format(n_labels))
    click.echo("#############################################")
