#!/usr/bin/env python
"""Computes statistics on a BioGenerator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import click
import numpy as np
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption, log_parameters)
from bob.learn.tensorflow.dataset.bio import BioGenerator

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand,
    epilog="""\b
An example configuration could be::
    # define the database:
    from bob.bio.base.test.dummy.database import database
    groups = ['dev']
    biofiles = database.all_files(groups)
"""
)
@click.option(
    '--database',
    '-d',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.bio.database',
    help='A bio database. Its original_directory must point to the correct '
    'path.')
@click.option(
    '--biofiles',
    required=True,
    cls=ResourceOption,
    help='The list of the bio files. You can only provide this through '
    'config files.')
@click.option(
    '--load-data',
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.load_data',
    help='A callable with the signature of '
    '``data = load_data(database, biofile)``. '
    ':any:`bob.bio.base.read_original_data` is used by default.')
@click.option(
    '--multiple-samples',
    is_flag=True,
    cls=ResourceOption,
    help='If provided, it means that the data provided by reader contains '
    'multiple samples with same label and path.')
@verbosity_option(cls=ResourceOption)
def compute_statistics(database, biofiles, load_data, multiple_samples,
                       **kwargs):
    """Computes statistics on a BioGenerator.

    This script works with bob.bio.base databases. It will load all the samples
    and print their mean.
    """
    log_parameters(logger, ignore=('biofiles', ))
    logger.debug("len(biofiles): %d", len(biofiles))

    assert len(biofiles), "biofiles are empty!"
    logger.info('Calculating the mean for %d files', len(biofiles))

    generator = BioGenerator(
        database,
        biofiles,
        load_data=load_data,
        multiple_samples=multiple_samples)

    for i, (data, _, _) in enumerate(generator()):
        if i == 0:
            mean = np.cast['float'](data)
        else:
            mean += data

    mean = mean.reshape(mean.shape[0], -1)
    mean = np.mean(mean, axis=1)
    click.echo(mean / (i + 1.))
