"""Convert datasets to TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import click

from bob.extension.scripts.click_helper import ConfigCommand
from bob.extension.scripts.click_helper import ResourceOption
from bob.extension.scripts.click_helper import verbosity_option

logger = logging.getLogger(__name__)


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

    To use this script with SGE, change your dataset (like shard it) and output a part
    of the dataset based on the SGE_TASK_ID environment variable in your config file.
    """
    import os

    from bob.extension.scripts.click_helper import log_parameters
    from bob.learn.tensorflow.data.tfrecords import dataset_to_tfrecord
    from bob.learn.tensorflow.data.tfrecords import tfrecord_name_and_json_name

    log_parameters(logger)

    output, json_output = tfrecord_name_and_json_name(output)
    if not force and os.path.isfile(output):
        click.echo("Output file already exists: {}".format(output))
        return

    click.echo("Writing tfrecod to: {}".format(output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    try:
        dataset_to_tfrecord(dataset, output)
    except Exception:
        click.echo("Something failed. Deleting unfinished files.")
        os.remove(output)
        os.remove(json_output)
        raise
    click.echo("Successfully wrote all files.")
