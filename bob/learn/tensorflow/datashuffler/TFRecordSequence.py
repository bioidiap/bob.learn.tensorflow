#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
import bob.ip.base
import numpy
from bob.learn.tensorflow.datashuffler.TFRecord import TFRecord


class TFRecordSequence(TFRecord):
    def __init__(self, filename_queue,
                 input_shape=[None, 28, 28, 1],
                 input_dtype="float32",
                 batch_size=32,
                 seed=10,
                 prefetch_capacity=50,
                 prefetch_threads=5,
                 sliding_win_len=5,
                 sliding_win_step=1,
                 min_after_dequeue=1):

        super(TFRecordSequence, self).__init__(filename_queue, input_shape, input_dtype,
                                               batch_size, seed, prefetch_capacity, prefetch_threads)

        self.min_after_dequeue = min_after_dequeue
        self.sliding_win_len = sliding_win_len
        self.sliding_win_step = sliding_win_step
        # sliding_win_shape = [-1, sliding_win_len] + list(input_shape[2:])
        self.sliding_win_shape = [-1, self.sliding_win_len] + list(self.input_shape[2:])
        # we assume that the second dimension is the temporal axis
        # so, the input data is of shape (batch_size, temporal_size, features, ....)
        # self.temporal_sequence_length = input_shape[1]
        #
        # # pre-compute number and shape of sliding windows
        # num_sliding_wins = (temporal_sequence_length - sliding_win_len) // sliding_win_step + 1
        # max_win_index = temporal_sequence_length - sliding_win_len + 1  # used in range function

    def generate_sliding_wins(self, data_ph, label_ph):
        """
        Assuming, the input is a temporal sequence, create a set of sliding windows
        :param data_ph: placeholder for data of input shape (batch_size, temporal_size, features_size, ...)
        :param label_ph: placeholder for label, assumed to be constant
        :return: sliding windows generated from sequence in data_ph and the corresponding array of labels
        """

        # Generate sliding windows
        print('inputs', data_ph)
        # we assume that the second dimension is the temporal axis
        # so, the input data is of shape (batch_size, temporal_size, features, ....)
        temporal_sequence_length = tf.shape(data_ph)[1]
        # pre-compute number and shape of sliding windows
        num_sliding_wins = (temporal_sequence_length - self.sliding_win_len) // self.sliding_win_step + 1
        max_win_index = temporal_sequence_length - self.sliding_win_len + 1  # used in range function

        # create sliding windows
        # data_sliding_wins = tf.map_fn(lambda i: data_ph[:, i:i + sliding_win_len], tf.range(0, max_win_index, sliding_win_step), dtype=tf.float32)
        # note that map_fn first applies lamda operation for i=0 to all batches, then for i=1, etc.,
        # so the final order of sliding window is not the one we need
        # we want first all windows for zeroth batch, then for first batch, etc. Hence, we re-arrange them later
        data_sliding_wins = tf.map_fn(lambda i: data_ph[:, i:i + self.sliding_win_len],
                                      tf.range(0, max_win_index, self.sliding_win_step),
                                      dtype=tf.float32)
        # make them of a correct shape
        # data_sliding_wins = tf.reshape(data_sliding_wins, sliding_win_shape)
        data_sliding_wins = tf.reshape(data_sliding_wins, self.sliding_win_shape)
        # since tf.map_fn returns sliding windows in a wrong order, we need to re-arrange the windows
        # we take even blocks first, then odd, and then concatenate them
        # data_sliding_wins_even = data_sliding_wins[::2]
        # data_sliding_wins_odd = data_sliding_wins[1::2]
        # data_sliding_wins = tf.concat([data_sliding_wins_even, data_sliding_wins_odd], 0)
        print('data_sliding_wins: ', data_sliding_wins)

        # correctly duplicate labels (the three operations emulate what numpy.repeat does)
        # label_sliding_win = tf.reshape(label_ph, [-1, 1])  # convert to len(label_ph) x 1 matrix.
        label_sliding_win = tf.tile(label_ph, [num_sliding_wins])  # Create multiple columns.
        # label_sliding_win = tf.tile(label_sliding_win, [1, num_sliding_wins])  # Create multiple columns.
        # label_sliding_win = tf.reshape(label_sliding_win, [-1])  # Convert back to a vector.
        print('label_sliding_win: ', label_sliding_win)

        return data_sliding_wins, label_sliding_win

    def create_placeholders(self):

        feature = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}

        # Define a reader and read the next record
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(self.filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the data from string back to the numbers
        data = tf.decode_raw(features['train/data'], tf.float32)
        print("data: ", data)
        # Cast label into int32
        label = tf.cast(features['train/label'], tf.int64)
        print("label: ", label)

        # Reshape data into the original shape
        data = tf.reshape(data, self.input_shape[1:])

        # get the placeholders from shuffle_batch
        data_ph, label_ph = tf.train.shuffle_batch([data, label],
                                                   batch_size=self.batch_size,
                                                   capacity=self.prefetch_capacity,
                                                   num_threads=self.prefetch_threads,
                                                   min_after_dequeue=self.min_after_dequeue,
                                                   name="shuffle_batch", seed=self.seed
                                                   )

        # data_ph, label_ph = tf.train.batch([data, label], batch_size=self.batch_size,
        #                                            capacity=self.prefetch_capacity, num_threads=self.prefetch_threads,
        #                                            name="simple_batch", enqueue_many=False)

        print("data_ph: ", data_ph)
        print("label_ph: ", label_ph)

        self.data_ph, self.label_ph = self.generate_sliding_wins(data_ph, label_ph)
