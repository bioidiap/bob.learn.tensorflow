#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import bob.ip.base
import numpy
from .DataAugmentation import DataAugmentation


class ImageAugmentation(DataAugmentation):
    """
    Class for applying common real-time random data augmentation for images.
    """

    def __init__(self, seed=10):

        super(ImageAugmentation, self).__init__(seed=seed)

        self.filter_bank = [self.__add_none,
                            self.__add_none,
                            self.__add_gaussian_blur,
                            self.__add_left_right_flip,
                            self.__add_none,
                            self.__add_salt_and_pepper]
    #self.__add_rotation,

    def __add_none(self, image):
        return image

    def __add_gaussian_blur(self, image):
        possible_sigmas = numpy.arange(0.1, 3., 0.1)
        possible_radii = [1, 2, 3]

        sigma = possible_sigmas[numpy.random.randint(len(possible_sigmas))]
        radius = possible_radii[numpy.random.randint(len(possible_radii))]

        gaussian_filter = bob.ip.base.Gaussian(sigma=(sigma, sigma),
                                               radius=(radius, radius))

        return gaussian_filter(image)

    def __add_left_right_flip(self, image):
        return bob.ip.base.flop(image)

    def __add_rotation(self, image):
        possible_angles = numpy.arange(-15, 15, 0.5)
        angle = possible_angles[numpy.random.randint(len(possible_angles))]

        return bob.ip.base.rotate(image, angle)

    def __add_salt_and_pepper(self, image):
        possible_levels = numpy.arange(0.01, 0.1, 0.01)
        level = possible_levels[numpy.random.randint(len(possible_levels))]

        return self.compute_salt_and_peper(image, level)

    def compute_salt_and_peper(self, image, level):
        """
        Compute a salt and pepper noise
        """
        r = numpy.random.rand(*image.shape)

        # 0 noise
        indexes_0 = r <= (level/0.5)
        image[indexes_0] = 0.0

        # 255 noise
        indexes_255 = (1 - level / 2) <= r;
        image[indexes_255] = 255.0

        return image
