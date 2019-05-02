#!/usr/bin/env python
"""Converts Bio and PAD datasets to TFRecords file formats.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import random
import tempfile
import click
import tensorflow as tf
from bob.io.base import create_directories_safe, HDF5File
from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    log_parameters,
)
from bob.learn.tensorflow.dataset.tfrecords import (
    describe_tf_record,
    write_a_sample,
    normalize_tfrecords_path,
    tfrecord_name_and_json_name,
    dataset_to_tfrecord,
)
from bob.learn.tensorflow.utils import bytes2human


logger = logging.getLogger(__name__)


@click.command(entry_point_group="bob.learn.tensorflow.config", cls=ConfigCommand)
@click.option(
    "--samples",
    required=True,
    cls=ResourceOption,
    help="A list of all samples that you want to write in the "
    "tfrecords file. Whatever is inside this list is passed to "
    "the reader.",
)
@click.option(
    "--reader",
    required=True,
    cls=ResourceOption,
    help="a function with the signature of ``data, label, key = "
    "reader(sample)`` which takes a sample and returns the "
    "loaded data, the label of the data, and a key which is "
    "unique for every sample.",
)
@click.option(
    "--output", "-o", required=True, cls=ResourceOption, help="Name of the output file."
)
@click.option(
    "--shuffle",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, it will shuffle the samples.",
)
@click.option(
    "--allow-failures",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, the samples which fail to load are ignored.",
)
@click.option(
    "--multiple-samples",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, it means that the data provided by reader contains "
    "multiple samples with same label and path.",
)
@click.option(
    "--size-estimate",
    is_flag=True,
    cls=ResourceOption,
    help="If given, will print the estimated file size instead of creating "
    "the final tfrecord file.",
)
@verbosity_option(cls=ResourceOption)
def db_to_tfrecords(
    samples,
    reader,
    output,
    shuffle,
    allow_failures,
    multiple_samples,
    size_estimate,
    **kwargs,
):
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
    log_parameters(logger, ignore=("samples",))
    logger.debug("len(samples): %d", len(samples))

    if size_estimate:
        output = tempfile.NamedTemporaryFile(suffix=".tfrecords").name

    output = normalize_tfrecords_path(output)

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
            logger.info("Processing file %d out of %d", i + 1, n_samples)

            data, label, key = reader(sample)

            if data is None:
                if allow_failures:
                    logger.debug("... Skipping `{0}`.".format(sample))
                    continue
                else:
                    raise RuntimeError("Reading failed for `{0}`".format(sample))

            if multiple_samples:
                for sample in data:
                    total_size += write_a_sample(
                        writer, sample, label, key, size_estimate=size_estimate
                    )
                    sample_count += 1
            else:
                total_size += write_a_sample(
                    writer, data, label, key, size_estimate=size_estimate
                )
                sample_count += 1

    if not size_estimate:
        click.echo("Wrote {} samples into the tfrecords file.".format(sample_count))
    else:
        # delete the empty tfrecords file
        try:
            os.remove(output)
        except Exception:
            pass
    click.echo(
        "The total size of the tfrecords file will be roughly "
        "{} bytes".format(bytes2human(total_size))
    )


@click.command()
@click.argument("tf-record-path", nargs=1)
@click.argument("shape", type=int, nargs=-1)
@click.option(
    "--batch-size", help="Batch size", show_default=True, required=True, default=1000
)
@verbosity_option(cls=ResourceOption)
def describe_tfrecord(tf_record_path, shape, batch_size, **kwargs):
    """
    Very often you have a tf-record file, or a set of them, and you have no
    idea how many samples you have there. Even worse, you have no idea how many
    classes you have.

    This click command will solve this thing for you by doing the following::

        $ %(prog)s <tf-record-path> 182 182 3

    """
    n_samples, n_labels = describe_tf_record(tf_record_path, shape, batch_size)
    click.echo("#############################################")
    click.echo("Number of samples {0}".format(n_samples))
    click.echo("Number of labels {0}".format(n_labels))
    click.echo("#############################################")


@click.command(entry_point_group="bob.learn.tensorflow.config", cls=ConfigCommand)
@click.option(
    "--dataset",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.dataset",
    help="A tf.data.Dataset to be used.",
)
@click.option(
    "--output", "-o", required=True, cls=ResourceOption, help="Name of the output file."
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    cls=ResourceOption,
    help="Whether to overwrite existing files.",
)
@verbosity_option(cls=ResourceOption)
def datasets_to_tfrecords(dataset, output, force, **kwargs):
    """Converts tensorflow datasets into TFRecords.
    Takes a list of datasets and outputs and writes each dataset into its output.
    ``datasets`` and ``outputs`` variables must be lists.
    You can convert the written TFRecord files back to datasets using
    :any:`bob.learn.tensorflow.dataset.tfrecords.dataset_from_tfrecord`.

    To use this script with SGE, change your dataset and output based on the SGE_TASK_ID
    environment variable in your config file.
    """
    log_parameters(logger)

    output, json_output = tfrecord_name_and_json_name(output)
    if not force and os.path.isfile(output):
        click.echo("Output file already exists: {}".format(output))
        return

    click.echo("Writing tfrecod to: {}".format(output))
    with tf.Session() as sess:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        try:
            sess.run(dataset_to_tfrecord(dataset, output))
        except Exception:
            click.echo("Something failed. Deleting unfinished files.")
            os.remove(output)
            os.remove(json_output)
            raise
    click.echo("Successfully wrote all files.")


@click.command(entry_point_group="bob.learn.tensorflow.config", cls=ConfigCommand)
@click.option(
    "--dataset",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.learn.tensorflow.dataset",
    help="A tf.data.Dataset to be used.",
)
@click.option(
    "--output", "-o", required=True, cls=ResourceOption, help="Name of the output file."
)
@click.option(
    "--mean",
    is_flag=True,
    cls=ResourceOption,
    help="If provided, the mean of data and labels will be saved in the hdf5 "
    'file as well. You can access them in the "mean" groups.',
)
@verbosity_option(cls=ResourceOption)
def dataset_to_hdf5(dataset, output, mean, **kwargs):
    """Saves a tensorflow dataset into an HDF5 file

    It is assumed that the dataset returns a tuple of (data, label, key) and
    the dataset is not batched.
    """
    log_parameters(logger)

    data, label, key = dataset.make_one_shot_iterator().get_next()

    sess = tf.Session()

    extension = ".hdf5"

    if not output.endswith(extension):
        output += extension

    create_directories_safe(os.path.dirname(output))

    sample_count = 0
    data_mean = 0.0
    label_mean = 0.0

    with HDF5File(output, "w") as f:
        while True:
            try:
                d, l, k = sess.run([data, label, key])
                group = "/{}".format(sample_count)
                f.create_group(group)
                f.cd(group)
                f["data"] = d
                f["label"] = l
                f["key"] = k
                sample_count += 1
                if mean:
                    data_mean += (d - data_mean) / sample_count
                    label_mean += (l - label_mean) / sample_count
            except tf.errors.OutOfRangeError:
                break
        if mean:
            f.create_group("/mean")
            f.cd("/mean")
            f["data_mean"] = data_mean
            f["label_mean"] = label_mean

    click.echo(f"Wrote {sample_count} samples into the hdf5 file.")
