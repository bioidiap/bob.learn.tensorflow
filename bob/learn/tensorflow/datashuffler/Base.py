#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
import bob.ip.base
import numpy


class Base(object):
    def __init__(self, data, labels,
                 input_shape,
                 input_dtype="float64",
                 scale=True,
                 batch_size=1,
                 seed=10,
                 data_augmentation=None):
        """
         The class provide base functionalities to shuffle the data before to train a neural network

         **Parameters**
           data:
           labels:
           perc_train:
           scale:
           train_batch_size:
           validation_batch_size:
           seed: Seed for the random number generator
        """
        self.seed = seed
        numpy.random.seed(seed)

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

        self.data_augmentation = data_augmentation

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

        skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))

        for i in range(bob_image.shape[0]):
            skimage[:, :, i] = bob_image[i, :, :]  # Copying red
        #skimage[:, :, 1] = bob_image[1, :, :]  # Copying green
        #skimage[:, :, 2] = bob_image[2, :, :]  # Copying blue

        return skimage

    def skimage2bob(self, sk_image):
        """
        Convert bob color image to the skcit image
        """

        bob_image = numpy.zeros(shape=(sk_image.shape[2], sk_image.shape[0], sk_image.shape[1]))

        for i in range(bob_image.shape[0]):
            bob_image[i, :, :] = sk_image[:, :, i]  # Copying red

        #bob_image[0, :, :] = sk_image[:, :, 0] # Copying red
        #if bob_image.shape[0] > 1:
        #    bob_image[1, :, :] = sk_image[:, :, 1]# Copying green
        #    bob_image[2, :, :] = sk_image[:, :, 2]  # Copying blue

        return bob_image



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
