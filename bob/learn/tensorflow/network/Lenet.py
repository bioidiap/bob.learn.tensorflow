#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the lenet architecture
"""

from ..util import *
import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv2D, FullyConnected, MaxPooling


class Lenet(SequenceNetwork):

    def __init__(self,
                 conv1_kernel_size=5,
                 conv1_output=16,

                 conv2_kernel_size=5,
                 conv2_output=32,

                 fc1_output=400,
                 n_classes=10,

                 seed=10, use_gpu = False):
        """
        Create all the necessary variables for this CNN

        **Parameters**
            conv1_kernel_size=5,
            conv1_output=32,

            conv2_kernel_size=5,
            conv2_output=64,

            fc1_output=400,
            n_classes=10

            seed = 10
        """
        super(Lenet, self).__init__()

        self.add(Conv2D(name="conv1", kernel_size=conv1_kernel_size, filters=conv1_output, activation=tf.nn.tanh))
        self.add(MaxPooling(name="pooling1"))
        self.add(Conv2D(name="conv2", kernel_size=conv2_kernel_size, filters=conv2_output, activation=tf.nn.tanh))
        self.add(MaxPooling(name="pooling2"))
        self.add(FullyConnected(name="fc1", output_dim=fc1_output, activation=tf.nn.tanh))
        self.add(FullyConnected(name="fc2", output_dim=n_classes, activation=None))
