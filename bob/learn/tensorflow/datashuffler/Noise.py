#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Mon  8 May 09:30:15 CEST 2017

import numpy
import tensorflow as tf


class Noise(object):

    """
    This datashuffler implements a generator of noise to feed a GAN's generator/ 

     **Parameters**

     input_shape:
       The shape of the inputs

     input_dtype:
       The type of the data,

     batch_size:
       Batch size

     seed:
       The seed of the random number generator
    """

    def __init__(self,
                 input_shape,
                 input_dtype="float64",
                 batch_size=1,
                 seed=10,
                 ):

        self.input_shape = tuple(input_shape)
        self.shape = tuple([batch_size] + input_shape)
        self.input_dtype = input_dtype
        self.batch_size = batch_size

        # Setting the seed
        numpy.random.seed(seed)

        self.data_placeholder = None
        self.label_placeholder = None

    def make_data(self):
      self.data = np.random.uniform(-1., 1., size=[self.batch_size, self.input_shape])

    def get_placeholders(self, name=""):
      if self.data_placeholder is None:
        self.data_placeholder = tf.placeholder(tf.float32, shape=self.shape, name=name)

      if self.label_placeholder is None:
        self.label_placeholder = tf.placeholder(tf.int64, shape=self.shape[0])

      return self.data_placeholder

    def get_batch(self):
        """
        Get a random batch of noise input.
        The noise is drawn from a uniform distribution in interval [-1 1]

        ** Returns **

        data:
          Selected samples

        """
        noise = numpy.random.uniform(-1., 1., size=[self.batch_size, self.input_shape[0]])
        
        return noise 
