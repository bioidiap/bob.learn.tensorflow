#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from .Layer import Layer
from operator import mul


class Dropout(Layer):

    """
    Dropout
    """

    def __init__(self, name,
                 keep_prob=0.99,
                 seed=10.
                 ):
        """
        Constructor

        **Parameters**
        input: Layer input
        activation: Tensor Flow activation
        initialization: Initialization type
        use_gpu: Store data in the GPU
        seed: Seed for the Random number generation
        """

        super(Dropout, self).__init__(name=name)
        self.keep_prob = keep_prob
        self.seed = seed

    def create_variables(self, input_layer):
        self.input_layer = input_layer
        return

    def get_graph(self, training_phase=True):

        with tf.name_scope(str(self.name)):
            output = tf.nn.dropout(self.input_layer, self.keep_prob, name=self.name)
            return output
