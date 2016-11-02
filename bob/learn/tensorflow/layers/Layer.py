#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensorflow as tf
from bob.learn.tensorflow.initialization import Xavier
from bob.learn.tensorflow.initialization import Constant


class Layer(object):

    """
    Layer base class
    """

    def __init__(self, name,
                 activation=None,
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 batch_norm=False,
                 use_gpu=False):
        """
        Base constructor

        **Parameters**
          name: Name of the layer
          activation: Tensorflow activation operation (https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html)
          weights_initialization: Initialization for the weights
          bias_initialization: Initialization for the biases
          use_gpu: I think this is not necessary to explain
          seed: Initialization seed set in Tensor flow
        """
        self.name = name
        self.weights_initialization = weights_initialization
        self.bias_initialization = bias_initialization
        self.use_gpu = use_gpu
        self.batch_norm = batch_norm

        self.input_layer = None
        self.activation = activation

        # Batch normalization variables
        self.beta = None
        self.gamma = None
        self.batch_mean = None
        self.batch_var = None

    def create_variables(self, input_layer):
        NotImplementedError("Please implement this function in derived classes")

    def get_graph(self, training_phase=True):
        NotImplementedError("Please implement this function in derived classes")

    def variable_exist(self, var):
        return var in [v.name.split("/")[0] for v in tf.all_variables()]

    def batch_normalize(self, x, phase_train):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Variable, true indicates training phase
            scope:       string, variable scope
            affn:      whether to affn-transform outputs
        Return:
            normed:      batch-normalized maps
        Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
        """
        from tensorflow.python.ops import control_flow_ops

        name = "batch_norm_" + str(self.name)
        reuse = self.variable_exist(name)

        #if reuse:
            #import ipdb; ipdb.set_trace();

        with tf.variable_scope(name, reuse=reuse):

            phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
            n_out = int(x.get_shape()[-1])

            self.beta = tf.get_variable(name + '_beta',
                                        initializer=tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                                        trainable=True,
                                        dtype=x.dtype)

            self.gamma = tf.get_variable(name + '_gamma',
                                         initializer=tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                                         trainable=True,
                                         dtype=x.dtype)

            if len(x.get_shape()) == 2:
                self.batch_mean, self.batch_var = tf.nn.moments(x, [0], name='moments_{0}'.format(name))
            else:
                self.batch_mean, self.batch_var = tf.nn.moments(x, range(len(x.get_shape())-1), name='moments_{0}'.format(name))

            ema = tf.train.ExponentialMovingAverage(decay=0.9)

            def mean_var_with_update():
                ema_apply_op = ema.apply([self.batch_mean, self.batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(self.batch_mean), tf.identity(self.batch_var)

            mean, var = control_flow_ops.cond(phase_train,
                                                        mean_var_with_update,
                                                        lambda: (ema.average(self.batch_mean), ema.average(self.batch_var)),
                                                        name=name + "mean_var")

            normed = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, 1e-3)
        return normed
