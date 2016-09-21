#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the architecture presented in the paper:

Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to
face verification." 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05). Vol. 1. IEEE, 2005.


"""

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv2D, FullyConnected, MaxPooling
import bob.learn.tensorflow
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class Chopra(SequenceNetwork):

    def __init__(self,
                 conv1_kernel_size=7,
                 conv1_output=15,

                 conv2_kernel_size=6,
                 conv2_output=45,

                 conv3_kernel_size=5,
                 conv3_output=250,

                 fc6_output=50,
                 n_classes=40,
                 default_feature_layer="fc7",

                 seed=10,
                 use_gpu=False):
        """
        Create all the necessary variables for this CNN

        **Parameters**
            conv1_kernel_size=5,
            conv1_output=32,

            conv2_kernel_size=5,
            conv2_output=64,

            conv3_kernel_size=5,
            conv3_output=250,

            fc6_output=50,
            n_classes=10

            seed = 10
        """
        super(Chopra, self).__init__(default_feature_layer=default_feature_layer,
                                    use_gpu=use_gpu)

        self.add(Conv2D(name="conv1", kernel_size=conv1_kernel_size,
                        filters=conv1_output,
                        activation=tf.nn.tanh,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling1"))

        self.add(Conv2D(name="conv2", kernel_size=conv2_kernel_size,
                        filters=conv2_output,
                        activation=tf.nn.tanh,
                        weights_initialization=Xavier(seed=seed,  use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling2"))

        self.add(Conv2D(name="conv3", kernel_size=conv3_kernel_size,
                        filters=conv3_output,
                        activation=tf.nn.tanh,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(FullyConnected(name="fc6", output_dim=fc6_output,
                                activation=tf.nn.tanh,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu)
                                ))
        self.add(FullyConnected(name="fc7", output_dim=n_classes,
                                activation=None,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu)))
