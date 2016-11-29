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

    **Parameters**

    name: str
      The name of the layer

    activation:
      Tensor Flow activation

    weights_initialization: py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the weights

    bias_initialization: py:class:`bob.learn.tensorflow.initialization.Initialization`
      Initialization type for the biases

    batch_norm: bool
      Do batch norm?

    use_gpu: bool
      Store data in the GPU

    """

    def __init__(self, name,
                 activation=None,
                 weights_initialization=Xavier(),
                 bias_initialization=Constant(),
                 batch_norm=False,
                 use_gpu=False):

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
        Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177

        ** Parameters **
            x:           Tensor, 4D BHWD input maps
            phase_train:

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

    def get_varible_by_name(self, var):
        """
        Doing this because of that https://github.com/tensorflow/tensorflow/issues/1325
        """

        for v in tf.all_variables():
            if (len(v.name.split("/")) > 1) and (var in v.name.split("/")[1]):
                return v

        return None
