#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""

"""

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv2D, FullyConnected, MaxPooling
import bob.learn.tensorflow
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class Chopra(SequenceNetwork):
    """Class that creates the architecture presented in the paper:

    Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to
    face verification." 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05). Vol. 1. IEEE, 2005.

    This is modifield version of the original architecture.
    It is inspired on https://gitlab.idiap.ch/bob/xfacereclib.cnn/blob/master/lua/network.lua

    -- C1 : Convolutional, kernel = 7x7 pixels, 15 feature maps

    -- M2 : MaxPooling, 2x2

    -- HT : Hard Hyperbolic Tangent

    -- C3 : Convolutional, kernel = 6x6 pixels, 45 feature maps

    -- M4 : MaxPooling, 4x3

    -- HT : Hard Hyperbolic Tangent

    -- R  : Reshaping layer HT 5x5 => 25 (45 times; once for each feature map)

    -- L5 : Linear 25 => 250


    **Parameters**

        conv1_kernel_size:

        conv1_output:

        pooling1_size:

        conv2_kernel_size:

        conv2_output:

        pooling2_size

        fc1_output:

        seed:
    """
    def __init__(self,
                 conv1_kernel_size=7,
                 conv1_output=15,

                 pooling1_size=[1, 2, 2, 1],


                 conv2_kernel_size=6,
                 conv2_output=45,

                 pooling2_size=[1, 4, 3, 1],

                 fc1_output=250,
                 default_feature_layer="fc1",

                 seed=10,
                 use_gpu=False,
                 batch_norm=False):

        super(Chopra, self).__init__(default_feature_layer=default_feature_layer,
                                     use_gpu=use_gpu)

        self.add(Conv2D(name="conv1", kernel_size=conv1_kernel_size,
                        filters=conv1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu),
                        batch_norm=batch_norm
                        ))
        self.add(MaxPooling(name="pooling1", shape=pooling1_size, activation=tf.nn.relu, batch_norm=False))

        self.add(Conv2D(name="conv2", kernel_size=conv2_kernel_size,
                        filters=conv2_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed,  use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu),
                        batch_norm=batch_norm))
        self.add(MaxPooling(name="pooling2", shape=pooling2_size, activation=tf.nn.relu, batch_norm=False))

        self.add(FullyConnected(name="fc1", output_dim=fc1_output,
                                activation=None,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu), batch_norm=False))
