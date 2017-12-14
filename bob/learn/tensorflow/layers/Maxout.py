#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date:  Fri 04 Aug 2017 14:14:22 CEST

## MAXOUT IMPLEMENTED FOR TENSORFLOW

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops

from tensorflow.python.layers import base


def maxout(inputs, num_units, axis=-1, name=None):
    return MaxOut(num_units=num_units, axis=axis, name=name)(inputs)


class MaxOut(base.Layer):
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
        super(MaxOut, self).__init__(name=name, trainable=False, **kwargs)
        self.axis = axis
        self.num_units = num_units

    def call(self, inputs, training=False):
        inputs = ops.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        if self.axis is None:
            # Assume that channel is the last dimension
            self.axis = -1
        num_channels = shape[self.axis]
        if num_channels % self.num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(
                                 num_channels, self.num_units))
        shape[self.axis] = -1
        shape += [num_channels // self.num_units]

        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = gen_array_ops.shape(inputs)[i]

        outputs = math_ops.reduce_max(
            gen_array_ops.reshape(inputs, shape), -1, keep_dims=False)
        shape = outputs.get_shape().as_list()
        shape[self.axis] = self.num_units
        outputs.set_shape(shape)

        return outputs
