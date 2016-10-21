#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Mon 05 Sep 2016 16:35 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")

from .Initialization import Initialization
import tensorflow as tf


class Constant(Initialization):
    """
    Implements the constant initialization.
    This is usually used to initialize biases.

    This tip were extracted from here
    http://www.deeplearningbook.org/contents/optimization.html

    page: 302

    """

    def __init__(self, constant_value=0.1, use_gpu=False, seed=None):

        self.constant_value = constant_value
        super(Constant, self).__init__(seed=None, use_gpu=use_gpu)

    def __call__(self, shape, name, scope):
        initializer = tf.constant(self.constant_value, shape=shape)

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
