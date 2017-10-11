#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Sun 16 Oct 2016 14:32:36 CEST

import numpy


class DataAugmentation(object):
    """
    Base class for applying common real-time data augmentation.

    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined augmentation methods will be applied at training
    time only.
    """

    def __init__(self, seed=10):
        self.filter_bank = []
        numpy.random.seed(seed)

    def __call__(self, image):
        """
        Apply a random filter to and image
        """

        if len(self.filter_bank) <= 0:
            raise ValueError("There is not filters in the filter bank")

        filter = self.filter_bank[numpy.random.randint(len(self.filter_bank))]
        return filter(image)


