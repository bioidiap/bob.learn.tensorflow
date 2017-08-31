#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
import bob.ip.base
import numpy
from bob.learn.tensorflow.datashuffler.Normalizer import Linear


class TFRecord(object):
    """
     The class generate batches using tfrecord

     **Parameters**

     filename:
       Name of the tf record

     input_shape:
       The shape of the inputs

     input_dtype:
       The type of the data,

     batch_size:
       Batch size

     seed:
       The seed of the random number generator

     data_augmentation:
       The algorithm used for data augmentation. Look :py:class:`bob.learn.tensorflow.datashuffler.DataAugmentation`

     normalizer:
       The algorithm used for feature scaling. Look :py:class:`bob.learn.tensorflow.datashuffler.ScaleFactor`, :py:class:`bob.learn.tensorflow.datashuffler.Linear` and :py:class:`bob.learn.tensorflow.datashuffler.MeanOffset`
       
     prefetch:
        Do prefetch?
        
     prefetch_capacity:
        

    """

    def __init__(self, filename_queue,
                 input_shape=[None, 28, 28, 1],
                 input_dtype="float32",
                 batch_size=32,
                 seed=10,
                 prefetch_capacity=50,
                 prefetch_threads=5):

        # Setting the seed for the pseudo random number generator
        self.seed = seed
        numpy.random.seed(seed)

        self.input_dtype = input_dtype

        # TODO: Check if the bacth size is higher than the input data
        self.batch_size = batch_size

        # Preparing the inputs
        self.filename_queue = filename_queue
        self.input_shape = tuple(input_shape)

        # Prefetch variables
        self.prefetch_capacity = prefetch_capacity
        self.prefetch_threads = prefetch_threads

        # Preparing placeholders
        self.data_ph = None
        self.label_ph = None
        
        self.data_ph_from_queue = None
        self.label_ph_from_queue = None
        self.prefetch = False


    def create_placeholders(self):
        """
        Create place holder instances
        
        :return: 
        """

        feature = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(self.filename_queue)        
        
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['train/data'], tf.float32)
        
        # Cast label data into int32
        label = tf.cast(features['train/label'], tf.int64)
        
        # Reshape image data into the original shape
        image = tf.reshape(image, self.input_shape[1:])
                
        images, labels = tf.train.shuffle_batch([image, label], batch_size=32, capacity=1000, num_threads=1, min_after_dequeue=1)
        self.data_ph = images
        self.label_ph = labels

        self.data_ph_from_queue = self.data_ph
        self.label_ph_from_queue = self.label_ph



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
            if from_queue:
                return self.data_ph_from_queue
            else:
                return self.data_ph

        else:
            if from_queue:
                return self.label_ph_from_queue
            else:
                return self.label_ph


    def get_batch(self):
        """
        Shuffle the Memory dataset and get a random batch.

        ** Returns **

        data:
          Selected samples

        labels:
          Correspondent labels
        """

        pass

