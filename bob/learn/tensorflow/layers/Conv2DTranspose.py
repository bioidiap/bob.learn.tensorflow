#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import tensorflow as tf
from .Layer import Layer
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant

import bob.core.log
import logging
logger = logging.getLogger("bob.learn")

class Conv2DTranspose(Layer):

    """
    2D Tranposed Convolution (or Deconv)

    **Parameters**

    name: str
      The name of the layer

    activation:
     Tensor Flow activation

    output_shape: list
      Shape of the ouptut: [height, width, layers]

    kernel_size: int
      Size of the convolutional kernel

    stride:
      Shape of the stride

    weights_initialization: py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    bias_initialization: py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    batch_norm: bool
      Do batch norm?

    use_gpu: bool
      Store data in the GPU

    """

    def __init__(self, name, 
                 output_shape,
                 activation=None,
                 kernel_size=3,
                 stride=[1, 1, 1, 1],
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 batch_norm=False,
                 use_gpu=False
                 ):
        super(Conv2DTranspose, self).__init__(name=name,
                                              activation=activation,
                                              weights_initialization=weights_initialization,
                                              bias_initialization=bias_initialization,
                                              batch_norm=batch_norm,
                                              use_gpu=use_gpu,
                                              )
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.W = None
        self.b = None
        self.stride = stride
        
        logger.info("+ adding a Conv2DTranspose layer ({0}) +".format(name))
        logger.info("\t kernel size = {0}".format(kernel_size))
        logger.info("\t stride = [{0}, {1}]".format(stride[1], stride[2]))
        logger.info("\t output dimension = {0}".format(output_shape))

    def create_variables(self, input_layer, scope="None"):
      # get the scope as "network_name/layer_name"
      if scope is not None:
        scope = scope + '/' + self.name
      else:
        scope = self.name
     
      self.scope = scope
      self.input_layer = input_layer

      # TODO: Do an assert here
      if len(input_layer.get_shape().as_list()) != 4:
        raise ValueError("The input of a transpose convolutional layer must have 4 dimensions, "
                             "but {0} were provided".format(len(input_layer.get_shape().as_list())))
      n_channels = input_layer.get_shape().as_list()[3]

      if self.W is None:

        variable = "weights"
        if self.get_varible_by_name(variable) is not None:
          self.W = self.get_varible_by_name(variable)
        else:
          self.W = self.weights_initialization(shape=[self.kernel_size, self.kernel_size, self.output_shape[-1], n_channels],
                                               name=variable,
                                               scope=scope
                                              )

        variable = "biases"
        if self.get_varible_by_name(variable) is not None:
          self.b = self.get_varible_by_name(variable)
        else:
          self.b = self.bias_initialization(shape=[self.output_shape[-1]],
                                            name=variable,
                                            scope=scope)
      
        logger.info("== Creating variables in layer {0}".format(self.name))
        logger.info("scope --> {0}".format(scope))
        logger.info("weights {0}".format([self.kernel_size, self.kernel_size, self.output_shape[-1], n_channels]))
        logger.info("biases {0}".format(self.output_shape[-1]))


    def get_graph(self, training_phase=True):

        with tf.name_scope(str(self.name)):
            batch_size = self.input_layer.get_shape().as_list()[0]
            self.output_shape.insert(0, batch_size)
            deconv2d = tf.nn.conv2d_transpose(self.input_layer, self.W, self.output_shape, strides=self.stride)
            self.output_shape.pop(0)
            # === This is done in the "original" code, but I don't see the point
            #deconv2d = tf.reshape(tf.nn.bias_add(deconv2d, self.b), deconv2d.get_shape())

            # batch_norm before or after adding the biases ???
            tf.nn.bias_add(deconv2d, self.b)

            if self.batch_norm:
                deconv2d = self.batch_normalize(deconv2d, training_phase, scope=self.scope)

            if self.activation is not None:
                output = self.activation(deconv2d)
            else:
                output = deconv2d

            return output
