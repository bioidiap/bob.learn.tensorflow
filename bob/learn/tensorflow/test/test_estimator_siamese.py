#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from nose.plugins.attrib import attr
import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Siamese, Logits

from bob.learn.tensorflow.dataset.siamese_image import shuffle_data_and_labels_image_augmentation as siamese_batch
from bob.learn.tensorflow.dataset.image import shuffle_data_and_labels_image_augmentation as single_batch

from bob.learn.tensorflow.loss import contrastive_loss, mean_cross_entropy_loss
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from .test_estimator_transfer import dummy_adapted
from bob.learn.tensorflow.utils import reproducible

import pkg_resources
import shutil


# Fixing problem with MAC https://github.com/dmlc/xgboost/issues/1715
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"
model_dir = "./temp"
model_dir_adapted = "./temp2"

learning_rate = 0.0001
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
# def test_siamesetrainer():
#     # Trainer logits
#     try:

#         # Setting seed
#         session_config, run_config, _, _, _ = reproducible.set_seed()
#         run_config = run_config.replace(save_checkpoints_steps=500)

#         trainer = Siamese(
#             model_dir=model_dir,
#             architecture=dummy,
#             config=run_config,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             loss_op=contrastive_loss,
#             validation_batch_size=validation_batch_size)
#         run_siamesetrainer(trainer)
#     finally:
#         try:
#             shutil.rmtree(model_dir, ignore_errors=True)
#             # pass
#         except Exception:
#             pass

# @attr('slow')
# def test_siamesetrainer_transfer():
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
#         # Setting seed
#         session_config, run_config, _, _, _ = reproducible.set_seed()
#         run_config = run_config.replace(save_checkpoints_steps=500)

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
#             config=run_config,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=False,
#             validation_batch_size=validation_batch_size)
#         logits_trainer.train(logits_input_fn, steps=steps)

#         # NOW THE FUCKING SIAMESE
#         trainer = Siamese(
#             model_dir=model_dir_adapted,
#             architecture=dummy_adapted,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             config=run_config,
#             loss_op=contrastive_loss,
#             validation_batch_size=validation_batch_size,
#             extra_checkpoint=extra_checkpoint)
#         run_siamesetrainer(trainer)
#     finally:
#         try:
#             shutil.rmtree(model_dir, ignore_errors=True)
#             shutil.rmtree(model_dir_adapted, ignore_errors=True)
#         except Exception:
#             pass

# @attr('slow')
# def test_siamesetrainer_transfer_extraparams():
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
#             "trainable_variables": ["Dummy"]
#         }

#         # Setting seed
#         session_config, run_config, _, _, _ = reproducible.set_seed()
#         run_config = run_config.replace(save_checkpoints_steps=500)

#         # LOGISTS
#         logits_trainer = Logits(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=2,
#             config=run_config,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=False,
#             validation_batch_size=validation_batch_size)

#         logits_trainer.train(logits_input_fn, steps=steps)

#         # NOW THE FUCKING SIAMESE
#         trainer = Siamese(
#             model_dir=model_dir_adapted,
#             architecture=dummy_adapted,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             loss_op=contrastive_loss,
#             config=run_config,
#             validation_batch_size=validation_batch_size,
#             extra_checkpoint=extra_checkpoint)
#         run_siamesetrainer(trainer)
#     finally:
#         try:
#             shutil.rmtree(model_dir, ignore_errors=True)
#             shutil.rmtree(model_dir_adapted, ignore_errors=True)
#         except Exception:
#             pass


def run_siamesetrainer(trainer):
    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    def input_fn():
        return siamese_batch(
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

    trainer.train(input_fn, steps=1, hooks=hooks)

    acc = trainer.evaluate(input_validation_fn)
    assert acc['accuracy'] > 0.3

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
