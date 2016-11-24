#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 09 Nov 2016 13:55:22 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")

from .Initialization import Initialization
import tensorflow as tf


class Uniform(Initialization):
    """
    Implements Random Uniform initialization
    """

    def __init__(self, seed=10., use_gpu=False):

        super(Uniform, self).__init__(seed, use_gpu=use_gpu)

    def __call__(self, shape, name, scope, init_value=None):

        if init_value is None:
            init_value = shape[0]
        import math
        # We use init_value as normalization value, but it can be used differently in different initializations
        stddev = 1.0 / math.sqrt(init_value)  # RANDOM UNIFORM INITIALIZATION
        initializer = tf.random_uniform(shape,
                                        minval=-stddev,
                                        maxval=stddev,
                                        seed=self.seed)

        try:
            with tf.variable_scope(scope):
                if self.use_gpu:
                    with tf.device("/gpu:0"):
                        return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
                else:
                    with tf.device("/cpu"):
                        return tf.get_variable(name, initializer=initializer, dtype=tf.float32)

        except ValueError:
            with tf.variable_scope(scope, reuse=True):
                if self.use_gpu:
                    with tf.device("/gpu:0"):
                        return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
                else:
                    with tf.device("/cpu"):
                        return tf.get_variable(name, initializer=initializer, dtype=tf.float32)

