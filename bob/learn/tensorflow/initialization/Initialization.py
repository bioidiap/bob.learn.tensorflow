#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Mon 05 Sep 2016 16:35 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")
import tensorflow as tf


class Initialization(object):
    """
    Base function for initialization.
    """

    def __init__(self, seed=10., use_gpu=False):
        """
        Default constructor

        **Parameters**

         shape: Shape of the input vector
         seed: Seed for the pseudo random number generator
         use_gpu: Variable stored in the GPU
        """

        self.seed = seed
        self.use_gpu = use_gpu
        tf.set_random_seed(seed)

    def variable_exist(self, var):
        return var in [v.name.split("/")[0] for v in tf.all_variables()]

    def __call__(self, shape, name, scope):
        NotImplementedError("Please implement this function in derived classes")
