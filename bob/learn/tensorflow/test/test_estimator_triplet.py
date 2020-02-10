#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from nose.plugins.attrib import attr
import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Triplet, Logits
from bob.learn.tensorflow.dataset.triplet_image import shuffle_data_and_labels_image_augmentation as triplet_batch
from bob.learn.tensorflow.dataset.image import shuffle_data_and_labels_image_augmentation as single_batch

from bob.learn.tensorflow.loss import triplet_loss, mean_cross_entropy_loss
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.utils import reproducible
import pkg_resources
from .test_estimator_transfer import dummy_adapted

import shutil

# Fixing problem with MAC https://github.com/dmlc/xgboost/issues/1715
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"
model_dir = "./temp"
model_dir_adapted = "./temp2"

learning_rate = 0.001
data_shape = (250, 250, 3)  # size of atnt images
output_shape = (50, 50)
data_type = tf.float32
batch_size = 4
validation_batch_size = 2
epochs = 1
steps = 5000

# Data
filenames = [
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
]
labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# @attr('slow')
# def test_triplet_estimator():
#     # Trainer logits
#     try:
#         trainer = Triplet(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             loss_op=triplet_loss,
#             validation_batch_size=validation_batch_size)
#         run_triplet_estimator(trainer)
#     finally:
#         try:
#             shutil.rmtree(model_dir, ignore_errors=True)
#             # pass
#         except Exception:
#             pass

# @attr('slow')
# def test_triplettrainer_transfer():
#     def logits_input_fn():
#         return single_batch(
#             filenames,
#             labels,
#             data_shape,
#             data_type,
#             batch_size,
#             epochs=epochs,
#             output_shape=output_shape)

#     # Trainer logits first than siamese
#     try:

#         extra_checkpoint = {
#             "checkpoint_path": model_dir,
#             "scopes": dict({
#                 "Dummy/": "Dummy/"
#             }),
#             "trainable_variables": []
#         }

#         # LOGISTS
#         logits_trainer = Logits(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=2,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=False,
#             validation_batch_size=validation_batch_size)
#         logits_trainer.train(logits_input_fn, steps=steps)

#         # NOW THE FUCKING SIAMESE
#         trainer = Triplet(
#             model_dir=model_dir_adapted,
#             architecture=dummy_adapted,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             loss_op=triplet_loss,
#             validation_batch_size=validation_batch_size,
#             extra_checkpoint=extra_checkpoint)
#         run_triplet_estimator(trainer)
#     finally:
#         try:
#             shutil.rmtree(model_dir, ignore_errors=True)
#             shutil.rmtree(model_dir_adapted, ignore_errors=True)
#         except Exception:
#             pass


def run_triplet_estimator(trainer):
    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    def input_fn():
        return triplet_batch(
            filenames,
            labels,
            data_shape,
            data_type,
            batch_size,
            epochs=epochs,
            output_shape=output_shape,
            random_flip=True,
            random_brightness=True,
            random_contrast=True,
            random_saturation=True)

    def input_validation_fn():
        return single_batch(
            filenames,
            labels,
            data_shape,
            data_type,
            validation_batch_size,
            epochs=10,
            output_shape=output_shape)

    hooks = [
        LoggerHookEstimator(trainer, batch_size, 300),
        tf.train.SummarySaverHook(
            save_steps=1000,
            output_dir=model_dir,
            scaffold=tf.train.Scaffold(),
            summary_writer=tf.summary.FileWriter(model_dir))
    ]

    trainer.train(input_fn, steps=steps, hooks=hooks)

    acc = trainer.evaluate(input_validation_fn)
    assert acc['accuracy'] > 0.3

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
