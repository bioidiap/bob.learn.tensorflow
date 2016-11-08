#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Mon 07 Nov 2016 09:39:36 CET


class ScaleFactor(object):
    """
    Normalize a sample by a scale factor
    """

    def __init__(self, scale_factor=0.00390625):
        self.scale_factor = scale_factor

    def __call__(self, x):
        return x * self.scale_factor


class MeanOffset(object):
    """
    Normalize a sample by a mean offset
    """

    def __init__(self, mean_offset):
        self.mean_offset = mean_offset

    def __call__(self, x):
        for i in range(len(self.mean_offset)):
            x[:, :, i] = x[:, :, i] - self.mean_offset[i]

        return x


class Linear(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return x


