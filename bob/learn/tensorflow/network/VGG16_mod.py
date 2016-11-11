#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv2D, FullyConnected, MaxPooling, Dropout, AveragePooling
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class VGG16_mod(SequenceNetwork):
    """
    Class that creates the VGG16 architecture modified

    Reference: Deep Face recognition: https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

    Basically, was removed the `fc6` and `fc7` and was implemented the average pooling in the conv5 layer

    """

    def __init__(self,
                 # First convolutional block
                 conv1_1_kernel_size=3,
                 conv1_1_output=64,

                 conv1_2_kernel_size=3,
                 conv1_2_output=64,

                 # Second convolutional block
                 conv2_1_kernel_size=3,
                 conv2_1_output=128,

                 conv2_2_kernel_size=3,
                 conv2_2_output=128,

                 # Third convolutional block
                 conv3_1_kernel_size=3,
                 conv3_1_output=256,

                 conv3_2_kernel_size=3,
                 conv3_2_output=256,

                 conv3_3_kernel_size=3,
                 conv3_3_output=256,

                 # Forth convolutional block
                 conv4_1_kernel_size=3,
                 conv4_1_output=512,

                 conv4_2_kernel_size=3,
                 conv4_2_output=512,

                 conv4_3_kernel_size=3,
                 conv4_3_output=512,

                 # Forth convolutional block
                 conv5_1_kernel_size=3,
                 conv5_1_output=256,

                 conv5_2_kernel_size=3,
                 conv5_2_output=256,

                 conv5_3_kernel_size=3,
                 conv5_3_output=256,

                 n_classes=10,

                 default_feature_layer="fc8",

                 seed=10,

                 do_dropout=True,

                 use_gpu=False):

        super(VGG16_mod, self).__init__(default_feature_layer=default_feature_layer,
                                        use_gpu=use_gpu)

        # First convolutional block
        self.conv1_1_kernel_size = conv1_1_kernel_size
        self.conv1_1_output = conv1_1_output

        self.conv1_2_kernel_size = conv1_2_kernel_size
        self.conv1_2_output = conv1_2_output

        # Second convolutional block
        self.conv2_1_kernel_size = conv2_1_kernel_size
        self.conv2_1_output = conv2_1_output

        self.conv2_2_kernel_size = conv2_2_kernel_size
        self.conv2_2_output = conv2_2_output

        # Third convolutional block
        self.conv3_1_kernel_size = conv3_1_kernel_size
        self.conv3_1_output = conv3_1_output

        self.conv3_2_kernel_size = conv3_2_kernel_size
        self.conv3_2_output = conv3_2_output

        self.conv3_3_kernel_size = conv3_3_kernel_size
        self.conv3_3_output = conv3_3_output

        # Forth convolutional block
        self.conv4_1_kernel_size = conv4_1_kernel_size
        self.conv4_1_output = conv4_1_output

        self.conv4_2_kernel_size = conv4_2_kernel_size
        self.conv4_2_output = conv4_2_output

        self.conv4_3_kernel_size = conv4_3_kernel_size
        self.conv4_3_output = conv4_3_output

        # Forth convolutional block
        self.conv5_1_kernel_size = conv5_1_kernel_size
        self.conv5_1_output = conv5_1_output

        self.conv5_2_kernel_size = conv5_2_kernel_size
        self.conv5_2_output = conv5_2_output

        self.conv5_3_kernel_size = conv5_3_kernel_size
        self.conv5_3_output = conv5_3_output

        self.n_classes = n_classes

        # First convolutional
        self.add(Conv2D(name="conv1_1", kernel_size=conv1_1_kernel_size,
                        filters=conv1_1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(Conv2D(name="conv1_2", kernel_size=conv1_2_kernel_size,
                        filters=conv2_1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed,  use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling1", strides=[1, 2, 2, 1]))

        # Second convolutional
        self.add(Conv2D(name="conv2_1", kernel_size=conv2_1_kernel_size,
                        filters=conv2_1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv2_2", kernel_size=conv2_2_kernel_size,
                        filters=conv2_2_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling2", strides=[1, 2, 2, 1]))

        # Third convolutional
        self.add(Conv2D(name="conv3_1", kernel_size=conv3_1_kernel_size,
                        filters=conv3_1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv3_2", kernel_size=conv3_2_kernel_size,
                        filters=conv3_2_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv3_3", kernel_size=conv3_3_kernel_size,
                        filters=conv3_3_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling3", strides=[1, 2, 2, 1]))


        # Forth convolutional
        self.add(Conv2D(name="conv4_1", kernel_size=conv4_1_kernel_size,
                        filters=conv4_1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv4_2", kernel_size=conv4_2_kernel_size,
                        filters=conv4_2_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv4_3", kernel_size=conv4_3_kernel_size,
                        filters=conv4_3_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling4", strides=[1, 2, 2, 1]))

        # Fifth convolutional
        self.add(Conv2D(name="conv5_1", kernel_size=conv5_1_kernel_size,
                        filters=conv5_1_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv5_2", kernel_size=conv5_2_kernel_size,
                        filters=conv5_2_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv5_3", kernel_size=conv5_3_kernel_size,
                        filters=conv5_3_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(AveragePooling(name="pooling5", strides=[1, 2, 2, 1]))

        if do_dropout:
            self.add(Dropout(name="dropout", keep_prob=0.4))

        self.add(FullyConnected(name="fc8", output_dim=n_classes,
                                activation=None,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu)))
