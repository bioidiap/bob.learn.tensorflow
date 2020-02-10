from nose.plugins.attrib import attr
from bob.learn.tensorflow.estimators import Regressor
from tensorflow import keras
import tensorflow as tf
import tensorflow.contrib.slim as slim

# @attr('slow')
# def test_regressor():

#     boston_housing = keras.datasets.boston_housing
#     (train_data, train_labels), (test_data,
#                                  test_labels) = boston_housing.load_data()

#     mean = train_data.mean(axis=0)
#     std = train_data.std(axis=0)
#     train_data = (train_data - mean) / std
#     test_data = (test_data - mean) / std

#     def input_fn(mode):
#         if mode == tf.estimator.ModeKeys.TRAIN:
#             features, labels = train_data, train_labels
#         else:
#             features, labels, = test_data, test_labels
#         dataset = tf.data.Dataset.from_tensor_slices(
#             (features, labels, [str(x) for x in labels]))
#         dataset = dataset.batch(1)
#         if mode == tf.estimator.ModeKeys.TRAIN:
#             dataset = dataset.apply(
#                 tf.contrib.data.shuffle_and_repeat(len(labels), 2))
#         data, label, key = dataset.make_one_shot_iterator().get_next()
#         return {'data': data, 'key': key}, label

#     def train_input_fn():
#         return input_fn(tf.estimator.ModeKeys.TRAIN)

#     def eval_input_fn():
#         return input_fn(tf.estimator.ModeKeys.EVAL)

#     def architecture(data, mode, **kwargs):
#         endpoints = {}

#         with tf.variable_scope('DNN'):

#             name = 'fc1'
#             net = slim.fully_connected(data, 64, scope=name)
#             endpoints[name] = net

#             name = 'fc2'
#             net = slim.fully_connected(net, 64, scope=name)
#             endpoints[name] = net

#         return net, endpoints

#     estimator = Regressor(architecture)

#     estimator.train(train_input_fn)

#     list(estimator.predict(eval_input_fn))

#     evaluations = estimator.evaluate(eval_input_fn)

#     assert 'rmse' in evaluations
#     assert 'loss' in evaluations
