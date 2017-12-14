#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import numpy


def scale_factor(x, scale_factor=0.00390625):
    """
    Normalize a sample by a scale factor
    """
    return x * scale_factor


def mean_offset(x, mean_offset):
    """
    Normalize a sample by a mean offset
    """

    for i in range(len(mean_offset)):
        x[:, :, i] = x[:, :, i] - mean_offset[i]

    return x


def per_image_standarization(x):

    mean = numpy.mean(x)
    std = numpy.std(x)

    return (x - mean) / max(std, 1 / numpy.sqrt(numpy.prod(x.shape)))
