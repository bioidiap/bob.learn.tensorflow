#!/usr/bin/env python
"""Script that computes statistics for image.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import pkg_resources so that bob imports work properly:
import pkg_resources
import os
import logging
import click
import numpy
import bob.io.image  # to be able to load images
from bob.io.base import save, load
from bob.extension.scripts.click_helper import verbosity_option

logger = logging.getLogger(__name__)


def process_images(base_path, extension, shape):

    files = os.listdir(base_path)
    sum_data = numpy.zeros(shape=shape)
    logging.info("Processing {0}".format(base_path))
    count = 0
    for f in files:
        path = os.path.join(base_path, f)
        if os.path.isdir(path):
            c, s = process_images(path, extension, shape)
            count += c
            sum_data += s

        if os.path.splitext(path)[1] == extension:
            data = load(path)
            count += 1
            sum_data += data

    return count, sum_data


@click.command()
@click.argument('base_path')
@click.argument('output_file')
@click.option('--extension', default='.hdf5', show_default=True)
@verbosity_option()
def compute_statistics(base_path, output_file, extension, **kwargs):
    """Script that computes statistics for image.
    """
    logger.debug('base_path: %s', base_path)
    logger.debug('output_file: %s', output_file)
    logger.debug('extension: %s', extension)
    logger.debug('kwargs: %s', kwargs)

    # SHAPE = [3, 224, 224]
    SHAPE = [1, 64, 64]

    count, sum_data = process_images(base_path, extension, SHAPE)

    means = numpy.zeros(shape=SHAPE)
    for s in range(SHAPE[0]):
        means[s, ...] = sum_data[s, ...] / float(count)

    save(means, output_file)
    save(means[0, :, :].astype("uint8"), "xuxa.png")
