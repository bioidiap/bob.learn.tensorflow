#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Mon 05 Sep 2016 16:35 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")

from .Initialization import Initialization
import tensorflow as tf


class Gaussian(Initialization):
    """
    Implements Gaussian normalization
    """

    def __init__(self, mean=0.,
                 std=1.,
                 seed=10.,
                 use_gpu=False):

        self.mean = mean
        self.std = std
        super(Gaussian, self).__init__(seed, use_gpu=use_gpu)

    def __call__(self, shape, name):

        if len(shape) == 4:
            in_out = shape[0] * shape[1] * shape[2] + shape[3]
        else:
            in_out = shape[0] + shape[1]

        initializer = tf.truncated_normal(shape,
                                          mean=self.mean,
                                          stddev=self.std,
                                          seed=self.seed)

        if self.use_gpu:
            with tf.device("/gpu"):
                return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
        else:
            with tf.device("/cpu"):
                return tf.get_variable(name, initializer=initializer, dtype=tf.float32)


