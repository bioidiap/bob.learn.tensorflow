#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the lenet architecture
"""

from ..util import *
import tensorflow as tf
from bob.learn.tensorflow.network.BaseArchitecture import BaseArchitecture


class Lenet(BaseArchitecture):

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

        self.conv1_kernel_size = conv1_kernel_size,
        self.conv1_output = conv1_output,

        self.conv2_kernel_size = conv2_kernel_size,
        self.conv2_output = conv2_output,

        self.fc1_output = fc1_output,
        self.n_classes = n_classes,

        super(Lenet, self).__init__(seed=seed, use_gpu=use_gpu)

    def create_variables(self):

        # First convolutional
        self.W_conv1 = create_weight_variables([self.conv1_kernel_size, self.conv1_kernel_size, 1, self.conv1_output],
                                               seed=self.seed, name="W_conv1", use_gpu=self.use_gpu)
        self.b_conv1 = create_bias_variables([self.conv1_output], name="bias_conv1", use_gpu=self.use_gpu)

        # Second convolutional
        self.W_conv2 = create_weight_variables([self.conv2_kernel_size, self.conv2_kernel_size, self.conv1_output,
                                                self.conv2_output], seed=self.seed, name="W_conv2",
                                               use_gpu=self.use_gpu)
        self.b_conv2 = create_bias_variables([self.conv2_output], name="bias_conv2", use_gpu=self.use_gpu)

        # First fc
        self.W_fc1 = create_weight_variables([(28 // 4) * (28 // 4) * self.conv2_output, self.fc1_output],
                                             seed=self.seed, name="W_fc1", use_gpu=self.use_gpu)
        self.b_fc1 = create_bias_variables([self.fc1_output], name="bias_fc1", use_gpu=self.use_gpu)

        # Second FC fc
        self.W_fc2 = create_weight_variables([self.fc1_output, self.n_classes], seed=self.seed,
                                             name="W_fc2", use_gpu=self.use_gpu)
        self.b_fc2 = create_bias_variables([self.n_classes], name="bias_fc2", use_gpu=self.use_gpu)

        self.seed = self.seed

    def create_graph(self, data):

        # Creating the architecture
        # First convolutional
        with tf.name_scope('conv_1') as scope:
            conv1 = create_conv2d(data, self.W_conv1)

        with tf.name_scope('tanh_1') as scope:
            tanh_1 = create_tanh(conv1, self.b_conv1)

        # Pooling
        with tf.name_scope('pool_1') as scope:
            pool1 = create_max_pool(tanh_1)

        # Second convolutional
        with tf.name_scope('conv_2') as scope:
            conv2 = create_conv2d(pool1, self.W_conv2)

        with tf.name_scope('tanh_2') as scope:
            tanh_2 = create_tanh(conv2, self.b_conv2)

        # Pooling
        with tf.name_scope('pool_2') as scope:
            pool2 = create_max_pool(tanh_2)

        #if train:
            #pool2 = tf.nn.dropout(pool2, 0.5, seed=self.seed)

        # Reshaping all the convolved images to 2D to feed the FC layers
        # FC1
        with tf.name_scope('fc_1') as scope:
            pool_shape = pool2.get_shape().as_list()
            reshape = tf.reshape(pool2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            fc1 = tf.nn.tanh(tf.matmul(reshape, self.W_fc1) + self.b_fc1)

        # FC2
        with tf.name_scope('fc_2') as scope:
            fc2 = tf.matmul(fc1, self.W_fc2) + self.b_fc2

        return fc2
