#!/usr/bin/env python

from keras.models import Sequential
from keras.engine import InputLayer
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Reshape

from keras.datasets import mnist
from keras.utils import np_utils

from keras.utils.layer_utils import print_summary

import os
import copy
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras import callbacks as cbks
from keras import optimizers, objectives
from keras import metrics as metrics_module

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend for the time being, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


def images_to_tfrecord(images, labels, filename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """ Save data into TFRecord """
    if not os.path.isfile(filename):
        num_examples = images.shape[0]

        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print('tfrecord %s already exists' % filename)


def read_and_decode_recordinput(tf_glob, one_hot=True, classes=None, is_train=None,
                                batch_shape=[1000, 28, 28, 1], parallelism=1):
    """ Return tensor to read from TFRecord """
    print 'Creating graph for loading %s TFRecords...' % tf_glob
    with tf.variable_scope("TFRecords"):
        record_input = data_flow_ops.RecordInput(
            tf_glob, batch_size=batch_shape[0], parallelism=parallelism)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_shape[0], 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        progbar = Progbar(len(records_op))

        images = []
        labels = []
        for i, serialized_example in enumerate(records_op):
            progbar.update(i)
            with tf.variable_scope("parse_images", reuse=True):
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'label': tf.FixedLenFeature([], tf.int64),
                        'image_raw': tf.FixedLenFeature([], tf.string),
                    })
                img = tf.decode_raw(features['image_raw'], tf.uint8)
                img.set_shape(batch_shape[1] * batch_shape[2])
                img = tf.reshape(img, [1] + batch_shape[1:])

                img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

                label = tf.cast(features['label'], tf.int32)
                if one_hot and classes:
                    label = tf.one_hot(label, classes)

                images.append(img)
                labels.append(label)

        images = tf.parallel_stack(images, 0)
        labels = tf.parallel_stack(labels, 0)
        images = tf.cast(images, tf.float32)

        images = tf.reshape(images, shape=batch_shape)

        # StagingArea will store tensors
        # across multiple steps to
        # speed up execution
        images_shape = images.get_shape()
        labels_shape = labels.get_shape()
        copy_stage = data_flow_ops.StagingArea(
            [tf.float32, tf.float32],
            shapes=[images_shape, labels_shape])
        copy_stage_op = copy_stage.put(
            [images, labels])
        staged_images, staged_labels = copy_stage.get()

        return images, labels


def save_mnist_as_tfrecord():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    images_to_tfrecord(images=X_train, labels=y_train, filename='train.mnist.tfrecord')
    images_to_tfrecord(images=X_test, labels=y_test, filename='test.mnist.tfrecord')


def lstm_layers(x_train_input, n_hidden, n_drop_first, n_new_steps, n_classes):
    x = LSTM(n_hidden, input_shape=(28, 28), return_sequences=True)(x_train_input)
    x = Lambda(lambda k: k[:, n_drop_first:, :])(x)
    x = Reshape((n_hidden*n_new_steps,), input_shape=(n_new_steps, n_hidden))(x)
    x_train_out = Dense(n_classes, activation="softmax",
                        name='x_train_out')(x)

    return x_train_out



def main(argv=None):

    sess = tf.Session()
    K.set_session(sess)

    save_mnist_as_tfrecord()

    n_epochs = 2
    n_hidden = 32  # Inside the LSTM cell
    n_drop_first = 2  # Number of first output to drop after LSTM
    classes = 10
    parallelism = 10
    batch_size = 100
    batch_shape = [batch_size, 28, 28, 1]


    x_train_batch, y_train_batch = read_and_decode_recordinput(
        'train.mnist.tfrecord',
        one_hot=True,
        classes=classes,
        is_train=True,
        batch_shape=batch_shape,
        parallelism=parallelism)

    x_test_batch, y_test_batch = read_and_decode_recordinput(
        'test.mnist.tfrecord',
        one_hot=True,
        classes=classes,
        is_train=True,
        batch_shape=batch_shape,
        parallelism=parallelism)

    x_batch_shape = x_train_batch.get_shape().as_list()
    y_batch_shape = y_train_batch.get_shape().as_list()

    print("Train data {}".format(x_batch_shape))
    print("Train labels {}".format(y_batch_shape))

    x_train_input = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
    n_steps = batch_shape[0]
    n_new_steps = n_steps - n_drop_first

    x_train_out = lstm_layers(x_train_input, n_hidden, n_drop_first, n_new_steps, classes)

    y_train_in_out = Input(tensor=y_train_batch, batch_shape=y_batch_shape, name='y_labels')
    cce = categorical_crossentropy(y_train_batch, x_train_out)

    # LSTM
    model = Sequential(inputs=[x_train_input], outputs=[x_train_out])
    model.add_loss(cce)

    ######################################################################

    print_summary(model)

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    tensorboard = TensorBoard()
    # tensorboard disabled due to Keras bug
    model.fit(batch_size=batch_size,
              epochs=n_epochs)  # callbacks=[tensorboard])

    model.save_weights('saved_wt.h5')

    K.clear_session()

    # Second Session, pure Keras
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    x_test_inp = Input(batch_shape=(None,) + (X_test.shape[1:]))
    test_out = lstm_layers(x_test_inp, n_hidden, n_drop_first, n_new_steps, classes)
    test_model = Model(inputs=x_test_inp, outputs=test_out)

    test_model.load_weights('saved_wt.h5')
    test_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    test_model.summary()

    loss, acc = test_model.evaluate(X_test, np_utils.to_categorical(y_test), classes)
    print('\nTest accuracy: {0}'.format(acc))

# for layer in model.layers:
#     print("{} {}".format(layer.name, model.get_layer(layer.name).output.shape))


if __name__ == '__main__':
    main()
