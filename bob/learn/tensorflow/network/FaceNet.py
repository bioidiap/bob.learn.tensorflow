#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv2D, FullyConnected, MaxPooling
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class FaceNet(SequenceNetwork):
    """
    Class that creates the The FaceNet architecture used in

    Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
    "Facenet: A unified embedding for face recognition and clustering."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

    This CNN has an intriguing property; some layers use 2d convolution of 1x1.
    This concept is described in:

    Ahuja, Ravindra K., Thomas L. Magnanti, and James B. Orlin.
    "Network flows: theory, algorithms, and applications." (1993).

    """

    def __init__(self,
                 ### First macro layer
                 conv1_kernel_size=7,
                 conv1_output=64,
                 conv1_stride=[1, 2, 2, 1], #[1, 2, 2, 1],

                 pool1_shape=[1, 2, 2, 1],

                 ### Second macro layer
                 conv2a_kernel_size=1,
                 conv2a_output=64,
                 conv2_kernel_size=3,
                 conv2_output=192,

                 pool2_shape=[1, 2, 2, 1],

                 ### Third macro layer
                 conv3a_kernel_size=1,
                 conv3a_output=192,
                 conv3_kernel_size=3,
                 conv3_output=192,

                 pool3_shape=[1, 2, 2, 1],

                 ### Forth macro layer
                 conv4a_kernel_size=1,
                 conv4a_output=384,
                 conv4_kernel_size=3,
                 conv4_output=384,

                 ### Fifth macro layer
                 conv5a_kernel_size=1,
                 conv5a_output=256,
                 conv5_kernel_size=3,
                 conv5_output=256,

                 ### Sixth macro layer
                 conv6a_kernel_size=1,
                 conv6a_output=256,
                 conv6_kernel_size=3,
                 conv6_output=256,

                 pool6_shape=[1, 2, 2, 1],


                 fc1_output=128,
                 fc2_output=128,

                 fc7128_output=128,

                 default_feature_layer="fc7128",
                 seed=10,
                 use_gpu=False):
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
        super(FaceNet, self).__init__(default_feature_layer=default_feature_layer,
                                    use_gpu=use_gpu)

        self.add(Conv2D(name="conv1", kernel_size=conv1_kernel_size,
                        filters=conv1_output,
                        activation=tf.nn.relu,
                        stride=conv1_stride,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling1", shape=pool1_shape))

        ##########
        self.add(Conv2D(name="conv2a", kernel_size=conv2a_kernel_size,
                        filters=conv2a_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(Conv2D(name="conv2", kernel_size=conv2_kernel_size,
                        filters=conv2_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling2", shape=pool2_shape))

        ##########
        self.add(Conv2D(name="conv3a", kernel_size=conv3a_kernel_size,
                        filters=conv3a_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv3", kernel_size=conv3_kernel_size,
                        filters=conv3_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling3", shape=pool3_shape))

        ##########
        self.add(Conv2D(name="conv4a", kernel_size=conv4a_kernel_size,
                        filters=conv4a_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv4", kernel_size=conv4_kernel_size,
                        filters=conv4_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        ##########
        self.add(Conv2D(name="conv5a", kernel_size=conv5a_kernel_size,
                        filters=conv5a_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv5", kernel_size=conv5_kernel_size,
                        filters=conv5_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        ##########
        self.add(Conv2D(name="conv6a", kernel_size=conv6a_kernel_size,
                        filters=conv6a_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))

        self.add(Conv2D(name="conv6", kernel_size=conv6_kernel_size,
                        filters=conv6_output,
                        activation=tf.nn.relu,
                        weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                        bias_initialization=Constant(use_gpu=self.use_gpu)
                        ))
        self.add(MaxPooling(name="pooling6", shape=pool6_shape))

        self.add(FullyConnected(name="fc1", output_dim=fc1_output,
                                activation=tf.nn.tanh,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu)
                                ))

        self.add(FullyConnected(name="fc2", output_dim=fc2_output,
                                activation=tf.nn.tanh,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu)
                                ))

        self.add(FullyConnected(name="fc7128", output_dim=fc7128_output,
                                activation=None,
                                weights_initialization=Xavier(seed=seed, use_gpu=self.use_gpu),
                                bias_initialization=Constant(use_gpu=self.use_gpu)
                                ))
