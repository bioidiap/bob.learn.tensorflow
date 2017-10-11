#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import tensorflow as tf
from .utils import append_logits


class Chopra(object):
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
        
        n_classes: If None, no Fully COnnected layer with class output will be created

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
                 n_classes=None,
                 seed=10):

            self.conv1_kernel_size = conv1_kernel_size
            self.conv1_output = conv1_output
            self.pooling1_size = pooling1_size

            self.conv2_output = conv2_output
            self.conv2_kernel_size = conv2_kernel_size
            self.pooling2_size = pooling2_size

            self.fc1_output = fc1_output

            self.seed = seed
            self.n_classes = n_classes


    def __call__(self, inputs, reuse=False, end_point='logits'):
        slim = tf.contrib.slim

        end_points = dict()
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32, seed=self.seed)

        graph = slim.conv2d(inputs, self.conv1_output, self.conv1_kernel_size, activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='conv1',
                            reuse=reuse)
        end_points['conv1'] = graph
        
        graph = slim.max_pool2d(graph, self.pooling1_size, scope='pool1')
        end_points['pool1'] = graph

        graph = slim.conv2d(graph, self.conv2_output, self.conv2_kernel_size, activation_fn=tf.nn.relu,
                            stride=1,
                            weights_initializer=initializer,
                            scope='conv2', reuse=reuse)
        end_points['conv2'] = graph
        graph = slim.max_pool2d(graph, self.pooling2_size, scope='pool2')
        end_points['pool2'] = graph        

        graph = slim.flatten(graph, scope='flatten1')
        end_points['flatten1'] = graph        

        graph = slim.fully_connected(graph, self.fc1_output,
                                     weights_initializer=initializer,
                                     activation_fn=None,
                                     scope='fc1',
                                     reuse=reuse)
        end_points['fc1'] = graph                                     
                                     
        if self.n_classes is not None:
            # Appending the logits layer
            graph = append_logits(graph, self.n_classes, reuse)
            end_points['logits'] = graph
        
        return end_points[end_point]

