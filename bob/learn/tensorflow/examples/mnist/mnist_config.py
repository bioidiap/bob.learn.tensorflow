#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# create reproducible nets:
from bob.learn.tensorflow.utils.reproducible import run_config
import tensorflow as tf
from bob.db.mnist import Database

model_dir = '/tmp/mnist_model'
train_tfrecords = ['/tmp/mnist_data/train.tfrecords']
eval_tfrecords = ['/tmp/mnist_data/test.tfrecords']

run_config = run_config.replace(keep_checkpoint_max=10**3)
run_config = run_config.replace(save_checkpoints_secs=60)


def input_fn(mode, batch_size=1):
    """A simple input_fn using the contrib.data input pipeline."""

    def example_parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'key': tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features['data'], tf.uint8)
        image.set_shape([28 * 28])

        # Normalize the values of the image from the range
        # [0, 255] to [-0.5, 0.5]
        image = tf.cast(image, tf.float32) / 255 - 0.5
        label = tf.cast(features['label'], tf.int32)

        key = tf.cast(features['key'], tf.string)
        return image, tf.one_hot(label, 10), key

    if mode == tf.estimator.ModeKeys.TRAIN:
        tfrecords_files = train_tfrecords
    elif mode == tf.estimator.ModeKeys.EVAL:
        tfrecords_files = eval_tfrecords
    else:
        assert mode == tf.estimator.ModeKeys.PREDICT, 'invalid mode'
        tfrecords_files = eval_tfrecords

    for tfrecords_file in tfrecords_files:
        assert tf.gfile.Exists(tfrecords_file), (
            'Run github.com:tensorflow/models/official/mnist/'
            'convert_to_records.py first to convert the MNIST data to '
            'TFRecord file format.')

    dataset = tf.data.TFRecordDataset(tfrecords_files)

    # For training, repeat the dataset forever
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()

    # Map example_parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(
        example_parser, num_parallel_calls=1).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    images, labels, keys = dataset.make_one_shot_iterator().get_next()

    return {'images': images, 'keys': keys}, labels


def train_input_fn():
    return input_fn(tf.estimator.ModeKeys.TRAIN)


def eval_input_fn():
    return input_fn(tf.estimator.ModeKeys.EVAL)


def predict_input_fn():
    return input_fn(tf.estimator.ModeKeys.PREDICT)


def mnist_model(inputs, mode):
    """Takes the MNIST inputs and mode and outputs a tensor of logits."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    data_format = 'channels_last'

    if tf.test.is_built_with_cuda():
        # When running on GPU, transpose the data from channels_last (NHWC) to
        # channels_first (NCHW) to improve performance. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        data_format = 'channels_first'
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        data_format=data_format)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2, data_format=data_format)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        data_format=data_format)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2, data_format=data_format)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)
    return logits


def model_fn(features, labels=None, mode=tf.estimator.ModeKeys.TRAIN):
    """Model function for MNIST."""
    keys = features['keys']
    features = features['images']
    logits = mnist_model(features, mode)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'keys': keys,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss,
                                      tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    with tf.name_scope('train_metrics'):
        # Create a tensor named train_accuracy for logging purposes
        tf.summary.scalar('train_accuracy', accuracy[1])

        tf.summary.scalar('train_loss', loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


estimator = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, params=None, config=run_config)

output = train_tfrecords[0]
db = Database()
data, labels = db.data(groups='train')

# output = eval_tfrecords[0]
# db = Database()
# data, labels = db.data(groups='test')

samples = zip(data, labels, (str(i) for i in range(len(data))))


def reader(sample):
    return sample
