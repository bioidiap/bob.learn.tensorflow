#!/usr/bin/env python
"""Computes statistics on a BioGenerator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import click
import numpy as np
from bob.extension.scripts.click_helper import (verbosity_option,
                                                ConfigCommand, ResourceOption)
from bob.learn.tensorflow.dataset.bio import BioGenerator

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.option(
    '--database',
    '-d',
    required=True,
    cls=ResourceOption,
    entry_point_group='bob.bio.database')
@click.option(
    '--biofiles',
    required=True,
    cls=ResourceOption,
    help='You can only provide this through config files.')
@click.option(
    '--load-data',
    cls=ResourceOption,
    entry_point_group='bob.learn.tensorflow.load_data')
@click.option('--multiple-samples', is_flag=True, cls=ResourceOption)
@verbosity_option(cls=ResourceOption)
def compute_statistics(database, biofiles, load_data, multiple_samples,
                       **kwargs):
    """Computes statistics on a BioGenerator.

    This script works with bob.bio.base databases. It will load all the samples
    and print their mean.

    \b
    Parameters
    ----------
    database : :any:`bob.bio.base.database.BioDatabase`
        A bio database. Its original_directory must point to the correct path.
    biofiles : [:any:`bob.bio.base.database.BioFile`]
        The list of the bio files.
    load_data : callable, optional
        A callable with the signature of
        ``data = load_data(database, biofile)``.
        :any:`bob.bio.base.read_original_data` is used by default.
    multiple_samples : bool, optional
        If provided, it assumes that the db interface returns several samples
        from a biofile. This option can be used when you are working with
        sequences.
    verbose : int, optional
        Increases verbosity (see help for --verbose).

    \b
    [CONFIG]...            Configuration files. It is possible to pass one or
                           several Python files (or names of
                           ``bob.learn.tensorflow.config`` entry points or
                           module names) which contain the parameters listed
                           above as Python variables. The options through the
                           command-line (see below) will override the values of
                           configuration files.

    An example configuration could be::

        # define the database:
        from bob.bio.base.test.dummy.database import database

        groups = ['dev']
        biofiles = database.all_files(groups)
    """
    logger.debug('database: %s', database)
    logger.debug('len(biofiles): %s', len(biofiles))
    logger.debug('load_data: %s', load_data)
    logger.debug('multiple_samples: %s', multiple_samples)
    logger.debug('kwargs: %s', kwargs)

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
