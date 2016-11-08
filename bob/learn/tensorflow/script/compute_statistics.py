#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Script that computes statistics for image

Usage:
  compute_statistics.py <base_path> --extension=<arg>
  compute_statistics.py -h | --help
Options:
  -h --help     Show this screen.
  --extension=<arg>    [default: .hdf5]
"""

from docopt import docopt
import bob.io.base
import os
import numpy
import bob.io.image

def process_images(base_path, extension, shape):

    files = os.listdir(base_path)
    sum_data = numpy.zeros(shape=shape)
    count = 0
    for f in files:
        path = os.path.join(base_path, f)
        if os.path.isdir(path):
            c, s = process_images(path, extension, shape)
            count += c
            sum_data += s

        if os.path.splitext(path)[1] == extension:
            data = bob.io.base.load(path)
            count += 1
            sum_data += data

    return count, sum_data


def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BASE_PATH = args['<base_path>']
    EXTENSION = args['--extension']
    SHAPE = [3, 250, 250]

    count, sum_data = process_images(BASE_PATH, EXTENSION, SHAPE)

    means = numpy.zeros(shape=SHAPE)
    for s in range(SHAPE[0]):
        means[s, ...] = sum_data[s, ...] / float(count)

    bob.io.base.save(means, "means.hdf5")
