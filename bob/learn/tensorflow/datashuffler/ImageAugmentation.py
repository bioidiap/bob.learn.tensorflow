#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import bob.ip.base
import numpy


def add_gaussian_blur(image, seed=10):
    """
    Add random gaussian blur
    """
    numpy.random.seed(seed)

    possible_sigmas = numpy.arange(0.1, 3., 0.1)
    possible_radii = [1, 2, 3]

    sigma = possible_sigmas[numpy.random.randint(len(possible_sigmas))]
    radius = possible_radii[numpy.random.randint(len(possible_radii))]

    gaussian_filter = bob.ip.base.Gaussian(
        sigma=(sigma, sigma), radius=(radius, radius))

    return gaussian_filter(image)


def add_rotation(image):
    """
    Add random rotation
    """

    possible_angles = numpy.arange(-15, 15, 0.5)
    angle = possible_angles[numpy.random.randint(len(possible_angles))]

    return bob.ip.base.rotate(image, angle)


def add_salt_and_pepper(image):
    """
    Add random salt and pepper
    """

    possible_levels = numpy.arange(0.01, 0.1, 0.01)
    level = possible_levels[numpy.random.randint(len(possible_levels))]

    return compute_salt_and_peper(image, level)


def compute_salt_and_peper(image, level):
    """
    Compute a salt and pepper noise
    """
    r = numpy.random.rand(*image.shape)

    # 0 noise
    indexes_0 = r <= (level / 0.5)
    image[indexes_0] = 0.0

    # 255 noise
    indexes_255 = (1 - level / 2) <= r
    image[indexes_255] = 255.0

    return image
