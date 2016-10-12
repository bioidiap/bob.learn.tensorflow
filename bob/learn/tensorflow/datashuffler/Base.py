#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
import bob.ip.base


class Base(object):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1):
        """
         The class provide base functionoalies to shuffle the data

         **Parameters**
           data:
           labels:
           perc_train:
           scale:
           train_batch_size:
           validation_batch_size:
        """

        self.scale = scale
        self.scale_value = 0.00390625
        self.input_dtype = input_dtype

        # TODO: Check if the bacth size is higher than the input data
        self.batch_size = batch_size

        self.data = data
        self.shape = tuple([batch_size] + input_shape)
        self.input_shape = tuple(input_shape)

        self.labels = labels
        self.possible_labels = list(set(self.labels))

        # Computing the data samples fro train and validation
        self.n_samples = len(self.labels)

        # Shuffling all the indexes
        self.indexes = numpy.array(range(self.n_samples))
        numpy.random.shuffle(self.indexes)

        self.data_placeholder = None
        self.label_placeholder = None

    def get_placeholders(self, name=""):
        """
        Returns a place holder with the size of your batch
        """

        if self.data_placeholder is None:
            self.data_placeholder = tf.placeholder(tf.float32, shape=self.shape, name=name)

        if self.label_placeholder is None:
            self.label_placeholder = tf.placeholder(tf.int64, shape=self.shape[0])

        return [self.data_placeholder, self.label_placeholder]

    def get_placeholders_forprefetch(self, name=""):
        """
        Returns a place holder with the size of your batch
        """
        if self.data_placeholder is None:
            self.data_placeholder = tf.placeholder(tf.float32, shape=tuple([None] + list(self.shape[1:])), name=name)
            self.label_placeholder = tf.placeholder(tf.int64, shape=[None, ])
        return [self.data_placeholder, self.label_placeholder]

    def bob2skimage(self, bob_image):
        """
        Convert bob color image to the skcit image
        """

        skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], 3))

        skimage[:, :, 0] = bob_image[0, :, :]  # Copying red
        skimage[:, :, 1] = bob_image[1, :, :]  # Copying green
        skimage[:, :, 2] = bob_image[2, :, :]  # Copying blue

        return skimage

    def rescale(self, data):
        """
        Reescale a single sample with input_shape

        """
        # if self.input_shape != data.shape:
        if self.bob_shape != data.shape:

            # TODO: Implement a better way to do this reescaling
            # If it is gray scale
            if self.input_shape[2] == 1:
                copy = data[:, :, 0].copy()
                dst = numpy.zeros(shape=self.input_shape[0:2])
                bob.ip.base.scale(copy, dst)
                dst = numpy.reshape(dst, self.input_shape)
            else:
                # dst = numpy.resize(data, self.bob_shape) # Scaling with numpy, because bob is c,w,d instead of w,h,c
                dst = numpy.zeros(shape=self.bob_shape)

                # TODO: LAME SOLUTION
                if data.shape[0] != 3:  # GRAY SCALE IMAGES IN A RGB DATABASE
                    step_data = numpy.zeros(shape=(3, data.shape[0], data.shape[1]))
                    step_data[0, ...] = data[:, :]
                    step_data[1, ...] = data[:, :]
                    step_data[2, ...] = data[:, :]
                    data = step_data

                bob.ip.base.scale(data, dst)

            return dst
        else:
            return data