#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""

"""

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
#from bob.learn.tensorflow.initialization import Xavier
#from bob.learn.tensorflow.initialization import Constant

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
                 conv1_kernel_size=[7, 7],
                 conv1_output=15,

                 pooling1_size=[2, 2],


                 conv2_kernel_size=[6, 6],
                 conv2_output=45,

                 pooling2_size=[4, 3],

                 fc1_output=250,
                 default_feature_layer="fc1",

                 seed=10,
                 use_gpu=False,
                 batch_norm=False):

        slim = tf.contrib.slim
        graph = slim.conv2d(data_placeholder, conv1_output, conv1_kernel_size, activation_fn=tf.nn.relu,
                            stride=2, scope='conv1')
        graph = slim.max_pool2d(graph, pooling1_size, scope='pool1')

        graph = slim.conv2d(graph, conv2_output, conv2_kernel_size, activation_fn=tf.nn.relu,
                            stride=2, scope='conv2')
        graph = slim.max_pool2d(graph, pooling2_size, scope='pool2')

        graph = slim.flatten(graph, scope='flatten1')

        graph = slim.fully_connected(graph, fc1_output, scope='fc1')

        super(Chopra, self).__init__(default_feature_layer=default_feature_layer,
                                     use_gpu=use_gpu)
