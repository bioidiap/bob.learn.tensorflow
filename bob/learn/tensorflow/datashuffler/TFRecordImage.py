#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
import bob.ip.base
import numpy
from bob.learn.tensorflow.datashuffler.Normalizer import Linear
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

    def __init__(self,filename_queue,
                         input_shape=[None, 28, 28, 1],
                         output_shape=[None, 28, 28, 1],
                         input_dtype=tf.uint8,
                         batch_size=32,
                         seed=10,
                         prefetch_capacity=50,
                         prefetch_threads=5, 
                         shuffle=True,
                         normalization=False,
                         random_flip=True,
                         random_crop=True,
                         gray_scale=False
                         ):

        # Setting the seed for the pseudo random number generator
        self.seed = seed
        numpy.random.seed(seed)

        self.input_dtype = input_dtype

        # TODO: Check if the bacth size is higher than the input data
        self.batch_size = batch_size

        # Preparing the inputs
        self.filename_queue = filename_queue
        self.input_shape = tuple(input_shape)
        self.output_shape = output_shape

        # Prefetch variables
        self.prefetch = True
        self.prefetch_capacity = prefetch_capacity
        self.prefetch_threads = prefetch_threads
        
        self.data_ph = None
        self.label_ph = None
        
        self.shuffle = shuffle
        self.normalization = normalization
        self.random_crop = random_crop
        self.gray_scale = gray_scale

    def __call__(self, element, from_queue=False):
        """
        Return the necessary placeholder
        
        """

        if not element in ["data", "label"]:
            raise ValueError("Value '{0}' invalid. Options available are {1}".format(element, self.placeholder_options))

        # If None, create the placeholders from scratch
        if self.data_ph is None:
            self.create_placeholders()

        if element == "data":
            return self.data_ph
        else:
            return self.label_ph


    def create_placeholders(self):

        feature = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        
        _, serialized_example = reader.read(self.filename_queue)
        
        
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['train/data'], self.input_dtype)
        #image = tf.decode_raw(features['train/data'], tf.uint8)
        
        # Cast label data into int32
        label = tf.cast(features['train/label'], tf.int64)
        
        # Reshape image data into the original shape
        image = tf.reshape(image, self.input_shape[1:])

        # Casting to float32
        image = tf.cast(image, tf.float32)
        
        if self.gray_scale:
            image = tf.image.rgb_to_grayscale(image, name="rgb_to_gray")
            self.output_shape[3] = 1
        
        if self.random_crop:
            image = tf.image.resize_image_with_crop_or_pad(image, self.output_shape[1], self.output_shape[2])

        # normalizing data
        if self.normalization:
            image = tf.image.per_image_standardization(image)

        image.set_shape(tuple(self.output_shape[1:]))
        

        if self.shuffle:
            data_ph, label_ph = tf.train.shuffle_batch([image, label], batch_size=self.batch_size,
                             capacity=self.prefetch_capacity, num_threads=self.prefetch_threads,
                             min_after_dequeue=1, name="shuffle_batch")
        else:
            data_ph, label_ph = tf.train.batch([image, label], batch_size=self.batch_size,
                             capacity=self.prefetch_capacity, num_threads=self.prefetch_threads, name="batch")
        
        
        self.data_ph = data_ph
        self.label_ph = label_ph

