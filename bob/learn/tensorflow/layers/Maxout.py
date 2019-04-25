#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date:  Fri 04 Aug 2017 14:14:22 CEST

# MAXOUT IMPLEMENTED FOR TENSORFLOW

from tensorflow.python.layers import base
import tensorflow as tf


def maxout(inputs, num_units, axis=-1, name=None):
    return Maxout(num_units=num_units, axis=axis, name=name)(inputs)


class Maxout(base.Layer):
    """
     Adds a maxout op from

    "Maxout Networks"

    Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, Yoshua
    Bengio

    Usually the operation is performed in the filter/channel dimension. This can also be
    used after fully-connected layers to reduce number of features.

    **Parameters**
    inputs: Tensor input

    num_units: Specifies how many features will remain after maxout in the `axis` dimension (usually channel).
    This must be multiple of number of `axis`.

    axis: The dimension where max pooling will be performed. Default is the
      last dimension.

    name: Optional scope for name_scope.
    """

    def __init__(self, num_units, axis=-1, name=None, **kwargs):
        super(Maxout, self).__init__(name=name, trainable=False, **kwargs)
        self.axis = axis
        self.num_units = num_units

    def call(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tf.shape(inputs)[i]

        num_channels = shape[self.axis]
        if not isinstance(num_channels, tf.Tensor) and num_channels % self.num_units:
            raise ValueError(
                "number of features({}) is not "
                "a multiple of num_units({})".format(num_channels, self.num_units)
            )

        if self.axis < 0:
            axis = self.axis + len(shape)
        else:
            axis = self.axis
        assert axis >= 0, "Find invalid axis: {}".format(self.axis)

        expand_shape = shape[:]
        expand_shape[axis] = self.num_units
        k = num_channels // self.num_units
        expand_shape.insert(axis, k)

        outputs = tf.math.reduce_max(
            tf.reshape(inputs, expand_shape), axis, keepdims=False
        )
        return outputs
