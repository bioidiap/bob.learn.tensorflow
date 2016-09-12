#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

"""
Class that creates the lenet architecture
"""

import tensorflow as tf
from .SequenceNetwork import SequenceNetwork
from ..layers import Conv2D, FullyConnected, MaxPooling
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class MLP(SequenceNetwork):

    def __init__(self,
                 output_shape,
                 hidden_layers=[10],
                 hidden_activation=tf.nn.tanh,
                 output_activation=None,
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 use_gpu=False):
        """
        Create all the necessary variables for this CNN

        **Parameters**
            output_shape: Shape of the output
            hidden_layers: List that contains the amount of hidden layers, where each element is the number of neurons
            hidden_activation: Activation function of the hidden layer. If you set to `None`, the activation will be linear
            output_activation: Activation of the output layer.  If you set to `None`, the activation will be linear
            seed = 10
        """
        super(MLP, self).__init__(use_gpu=use_gpu)

        if (not (isinstance(hidden_layers, list) or isinstance(hidden_layers, tuple))) or len(hidden_layers) == 0:
            raise ValueError("Invalid input for hidden_layers: {0} ".format(hidden_layers))

        for i in range(len(hidden_layers)):
            l = hidden_layers[i]
            self.add(FullyConnected(name="fc{0}".format(i),
                                    output_dim=l,
                                    activation=hidden_activation,
                                    weights_initialization=weights_initialization,
                                    bias_initialization=bias_initialization))

        self.add(FullyConnected(name="fc_output",
                                output_dim=output_shape,
                                activation=output_activation,
                                weights_initialization=weights_initialization,
                                bias_initialization=bias_initialization))
