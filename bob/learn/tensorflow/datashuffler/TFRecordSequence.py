#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Thu 28 Sep 2017 11:35:22 CEST

import tensorflow as tf
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
        self.sliding_win_shape = [-1, self.sliding_win_len] + list(self.input_shape[2:])

    def generate_sliding_wins(self, data_ph, label_ph):
        """
        Assuming, the input is a temporal sequence, create a set of sliding windows
        :param data_ph: placeholder for data of input shape (batch_size, temporal_size, features_size, ...)
        :param label_ph: placeholder for label, assumed to be constant
        :return: sliding windows generated from sequence in data_ph and the corresponding array of labels
        """

        # we assume that the second dimension is the temporal axis
        # so, the input data is of shape (batch_size, temporal_size, features, ....)
        temporal_sequence_length = tf.shape(data_ph)[1]
        # pre-compute number and shape of sliding windows
        num_sliding_wins = (temporal_sequence_length - self.sliding_win_len) // self.sliding_win_step + 1
        max_win_index = temporal_sequence_length - self.sliding_win_len + 1  # used in range function

        # create sliding windows
        # note that map_fn first applies lamda operation for i=0 to all batches, then for i=1, etc.,
        data_sliding_wins = tf.map_fn(lambda i: data_ph[:, i:i + self.sliding_win_len],
                                      tf.range(0, max_win_index, self.sliding_win_step),
                                      dtype=tf.float32)
        # make them of a correct shape
        data_sliding_wins = tf.reshape(data_sliding_wins, self.sliding_win_shape)

        # correctly duplicate labels
        label_sliding_win = tf.tile(label_ph, [num_sliding_wins])

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
        # Cast label into int32
        label = tf.cast(features['train/label'], tf.int64)

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

        self.data_ph, self.label_ph = self.generate_sliding_wins(data_ph, label_ph)
