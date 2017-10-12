#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
import bob.ip.base
import numpy
import six


class Base(object):
    """
     The class provide base functionalities to shuffle the data to train a neural network

     **Parameters**

     data:
       Input data to be trainer

     labels:
       Labels. These labels should be set from 0..1

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

    def __init__(self, data, labels,
                 input_shape=[None, 28, 28, 1],
                 input_dtype="float32",
                 batch_size=32,
                 seed=10,
                 data_augmentation=None,
                 normalizer=None,
                 prefetch=False,
                 prefetch_capacity=50,
                 prefetch_threads=5):

        # Setting the seed for the pseudo random number generator
        self.seed = seed
        numpy.random.seed(seed)

        self.normalizer = normalizer
        self.input_dtype = input_dtype

        # TODO: Check if the bacth size is higher than the input data
        self.batch_size = batch_size

        # Preparing the inputs
        self.data = data
        self.input_shape = tuple(input_shape)
        self.labels = labels
        self.possible_labels = list(set(self.labels))

        # Computing the data samples fro train and validation
        self.n_samples = len(self.labels)

        # Shuffling all the indexes
        self.indexes = numpy.array(range(self.n_samples))
        numpy.random.shuffle(self.indexes)

        # Use data data augmentation?
        self.data_augmentation = data_augmentation

        # Preparing placeholders
        self.data_ph = None
        self.label_ph = None
        # Prefetch variables
        self.prefetch = prefetch
        self.prefetch_capacity = prefetch_capacity
        self.prefetch_threads = prefetch_threads
        self.data_ph_from_queue = None
        self.label_ph_from_queue = None

        self.batch_generator = None
        self.epoch = 0

    def create_placeholders(self):
        """
        Create place holder instances
        
        :return: 
        """
        with tf.name_scope("Input"):

            self.data_ph = tf.placeholder(tf.float32, shape=self.input_shape, name="data")
            self.label_ph = tf.placeholder(tf.int64, shape=[None], name="label")

            # If prefetch, setup the queue to feed data
            if self.prefetch:
                queue = tf.FIFOQueue(capacity=self.prefetch_capacity,
                                     dtypes=[tf.float32, tf.int64],
                                     shapes=[self.input_shape[1:], []])

                # Fetching the place holders from the queue
                self.enqueue_op = queue.enqueue_many([self.data_ph, self.label_ph])
                self.data_ph_from_queue, self.label_ph_from_queue = queue.dequeue_many(self.batch_size)

            else:
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


    def bob2skimage(self, bob_image):
        """
        Convert bob color image to the skcit image
        """

        skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))

        for i in range(bob_image.shape[0]):
            skimage[:, :, i] = bob_image[i, :, :]

        return skimage

    def skimage2bob(self, sk_image):
        """
        Convert bob color image to the skcit image
        """

        bob_image = numpy.zeros(shape=(sk_image.shape[2], sk_image.shape[0], sk_image.shape[1]))

        for i in range(bob_image.shape[0]):
            bob_image[i, :, :] = sk_image[:, :, i]  # Copying red

        return bob_image

    def rescale(self, data):
        """
        Reescale a single sample with input_shape

        """
        # if self.input_shape != data.shape:
        if self.bob_shape != data.shape:

            # TODO: Implement a better way to do this reescaling
            # If it is gray scale
            if self.input_shape[3] == 1:
                copy = data[:, :, 0].copy()
                dst = numpy.zeros(shape=self.input_shape[1:3])
                bob.ip.base.scale(copy, dst)
                dst = numpy.reshape(dst, self.input_shape[1:4])
            else:
                #dst = numpy.resize(data, self.bob_shape) # Scaling with numpy, because bob is c,w,d instead of w,h,c
                dst = numpy.zeros(shape=(data.shape[0], data.shape[1], 3))
                dst[:, :, 0] = data[:, :, 0]
                dst[:, :, 1] = data[:, :, 0]
                dst[:, :, 2] = data[:, :, 0]

                # TODO: LAME SOLUTION
                #if data.shape[0] != 3:  # GRAY SCALE IMAGES IN A RGB DATABASE
                #   step_data = numpy.zeros(shape=(3, data.shape[0], data.shape[1]))
                    #step_data = numpy.zeros(shape=(3, data.shape[0], data.shape[1]))
                    #step_data[0, ...] = data[:, :, 0]
                    #step_data[1, ...] = data[:, :, 0]
                    #step_data[2, ...] = data[:, :, 0]
                    #data = step_data
                #dst = numpy.zeros(shape=(self.bob_shape))
                #bob.ip.base.scale(data, dst)

            return dst
        else:
            return data

    def normalize_sample(self, x):
        """
        Normalize the sample.

        For the time being I'm only scaling from 0-1
        """

        if self.normalizer is None:
            return x
        else:
            return self.normalizer(x)

    def _aggregate_batch(self, data_holder, use_list=False):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in [int, bool]:
                    tp = 'int64'
                elif type(dt) == float:
                    tp = self.input_dtype
                else:
                    try:
                        tp = dt.dtype
                    except:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    result.append(
                        numpy.asarray([x[k] for x in data_holder], dtype=tp))
                except KeyboardInterrupt:
                    raise
                except:
                    #logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    import IPython as IP
                    IP.embed(config=IP.terminal.ipapp.load_default_config())
        return result

    def get_batch(self):
        """
        Shuffle the Memory dataset and get a random batch.

        ** Returns **

        data:
          Selected samples

        labels:
          Correspondent labels
        """

        if self.batch_generator is None:
            self.batch_generator = self._fetch_batch()

        holder = []
        try:
            for i in range(self.batch_size):
                data = six.next(self.batch_generator)
                
                holder.append(data)
                if len(holder) == self.batch_size:
                    return self._aggregate_batch(holder, False)

        except StopIteration:
            self.batch_generator = None
            self.epoch += 1
            
            # If we have left data in the epoch, return
            if len(holder) > 0:
                return self._aggregate_batch(holder, False)
            else:
                self.batch_generator = self._fetch_batch()
                data = six.next(self.batch_generator)
                holder.append(data)
                return self._aggregate_batch(holder, False)

