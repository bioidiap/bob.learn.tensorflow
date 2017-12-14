#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST

import numpy
import tensorflow as tf
import bob.ip.base
import numpy
from .TFRecord import TFRecord


class TFRecordImage(TFRecord):
    """
    Datashuffler that wraps the batching using tfrecords.

    This shuffler is more suitable for image datasets, because it does image data augmentation operations.

    **Parameters**

      filename_queue: Tensorflow producer
      input_shape: Shape of the input in the tfrecord
      output_shape: Desired output shape after the data augmentation
      input_dtype: Type of the raw data saved in the tf record
      batch_size: Size of the batch
      seed: Seed
      prefetch_capacity: Capacity of the bucket for prefetching
      prefetch_threads: Number of threads in the prefetching
      shuffle: Shuffled the batch
      normalization: zero mean and unit std
      random_flip:
      random_crop:
      gray_scale: Convert the output to gray scale
    """

    def __init__(self,
                 filename_queue,
                 input_shape=[None, 28, 28, 1],
                 output_shape=[None, 28, 28, 1],
                 input_dtype=tf.uint8,
                 batch_size=32,
                 seed=10,
                 prefetch_capacity=1000,
                 prefetch_threads=5,
                 shuffle=True,
                 normalization=False,
                 random_flip=True,
                 random_crop=True,
                 gray_scale=False):

        super(TFRecord, self).__init__(
            filename_queue=filename_queue,
            input_shape=input_shape,
            input_dtype=input_dtype,
            batch_size=batch_size,
            seed=seed,
            prefetch_capacity=prefetch_capacity,
            prefetch_threads=prefetch_threads)
        # Preparing the output
        self.output_shape = output_shape

        self.shuffle = shuffle
        self.normalization = normalization
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.gray_scale = gray_scale

    def create_placeholders(self):
        """
        Reimplementation
        """

        image, label = self.__load_features()

        # Casting to float32
        image = tf.cast(image, tf.float32)

        if self.gray_scale:
            image = tf.image.rgb_to_grayscale(image, name="rgb_to_gray")
            self.output_shape[3] = 1

        if self.random_crop:
            image = tf.image.resize_image_with_crop_or_pad(
                image, self.output_shape[1], self.output_shape[2])

        if self.random_flip:
            image = tf.image.random_flip_left_right(image)

        # normalizing data
        if self.normalization:
            image = tf.image.per_image_standardization(image)

        image.set_shape(tuple(self.output_shape[1:]))

        if self.shuffle:
            data_ph, label_ph = tf.train.shuffle_batch(
                [image, label],
                batch_size=self.batch_size,
                capacity=self.prefetch_capacity,
                num_threads=self.prefetch_threads,
                min_after_dequeue=self.prefetch_capacity // 2,
                name="shuffle_batch")
        else:
            data_ph, label_ph = tf.train.batch(
                [image, label],
                batch_size=self.batch_size,
                capacity=self.prefetch_capacity,
                num_threads=self.prefetch_threads,
                name="batch")

        self.data_ph = data_ph
        self.label_ph = label_ph
