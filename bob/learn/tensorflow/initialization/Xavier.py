#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Mon 05 Sep 2016 16:35 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")

from .Initialization import Initialization
import tensorflow as tf


class Xavier(Initialization):
    """
    Implements the classic and well used Xavier initialization as in

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Aistats. Vol. 9. 2010.


    Basically the initialization is Gaussian distribution with mean 0 and variance:

    Var(W) = 1/sqrt(n_{in} + n_{out});
    where n is the total number of parameters for input and output.
    """

    def __init__(self, seed=10., use_gpu=False):

        super(Xavier, self).__init__(seed, use_gpu=use_gpu)

    def __call__(self, shape, name):

        if len(shape) == 4:
            in_out = shape[0] * shape[1] * shape[2] + shape[3]
        else:
            in_out = shape[0] + shape[1]

        import math
        stddev = math.sqrt(3.0 / in_out)  # XAVIER INITIALIZER (GAUSSIAN)

        initializer = tf.truncated_normal(shape, stddev=stddev, seed=self.seed)

        if self.use_gpu:
            with tf.device("/gpu:0"):
                return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
        else:
            with tf.device("/cpu"):
                return tf.get_variable(name, initializer=initializer, dtype=tf.float32)


