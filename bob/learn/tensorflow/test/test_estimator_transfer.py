#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from nose.plugins.attrib import attr
import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Logits, LogitsCenterLoss
from bob.learn.tensorflow.utils import reproducible
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
from .test_estimator_onegraph import run_logitstrainer_mnist

import shutil
import os

# Fixing problem with MAC https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"
model_dir = "./temp"
model_dir_adapted = "./temp2"

learning_rate = 0.1
data_shape = (28, 28, 1)  # size of atnt images
data_type = tf.float32
batch_size = 32
validation_batch_size = 250
epochs = 6
steps = 5000


def dummy_adapted(inputs,
                  reuse=False,
                  mode=tf.estimator.ModeKeys.TRAIN,
                  trainable_variables=None,
                  **kwargs):
    """
    Create all the necessary variables for this CNN

    Parameters
    ----------
        inputs:

        reuse:

        mode:

        trainable_variables:
    """

    slim = tf.contrib.slim
    graph, end_points = dummy(
        inputs,
        reuse=reuse,
        mode=mode,
        trainable_variables=trainable_variables)

    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('Adapted', reuse=reuse):
        name = 'fc2'
        graph = slim.fully_connected(
            graph,
            50,
            weights_initializer=initializer,
            activation_fn=tf.nn.relu,
            scope=name,
            trainable=True)
        end_points[name] = graph

        name = 'fc3'
        graph = slim.fully_connected(
            graph,
            25,
            weights_initializer=initializer,
            activation_fn=None,
            scope=name,
            trainable=True)
        end_points[name] = graph

    return graph, end_points

# @attr('slow')
# def test_logitstrainer():
#     # Trainer logits
#     try:
#         _, run_config, _, _, _ = reproducible.set_seed()
#         embedding_validation = False
#         trainer = Logits(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             config=run_config)
#         run_logitstrainer_mnist(trainer, augmentation=True)
#         del trainer

#         ## Again
#         extra_checkpoint = {
#             "checkpoint_path": "./temp",
#             "scopes": dict({
#                 "Dummy/": "Dummy/"
#             }),
#             "trainable_variables": []
#         }

#         trainer = Logits(
#             model_dir=model_dir_adapted,
#             architecture=dummy_adapted,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             extra_checkpoint=extra_checkpoint,
#             config=run_config)

#         run_logitstrainer_mnist(trainer, augmentation=True)

#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#             shutil.rmtree(model_dir_adapted, ignore_errors=True)
#             pass
#         except Exception:
#             pass

# @attr('slow')
# def test_logitstrainer_center_loss():
#     # Trainer logits
#     try:
#         embedding_validation = False
#         _, run_config, _, _, _ = reproducible.set_seed()
#         trainer = LogitsCenterLoss(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             apply_moving_averages=False,
#             config=run_config)
#         run_logitstrainer_mnist(trainer, augmentation=True)
#         del trainer

#         ## Again
#         extra_checkpoint = {
#             "checkpoint_path": "./temp",
#             "scopes": dict({
#                 "Dummy/": "Dummy/"
#             }),
#             "trainable_variables": ["Dummy"]
#         }

#         trainer = LogitsCenterLoss(
#             model_dir=model_dir_adapted,
#             architecture=dummy_adapted,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             extra_checkpoint=extra_checkpoint,
#             apply_moving_averages=False,
#             config=run_config)

#         run_logitstrainer_mnist(trainer, augmentation=True)

#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#             shutil.rmtree(model_dir_adapted, ignore_errors=True)
#         except Exception:
#             pass
