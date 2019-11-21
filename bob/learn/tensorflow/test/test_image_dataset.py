#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Logits, LogitsCenterLoss

from bob.learn.tensorflow.dataset.image import shuffle_data_and_labels_image_augmentation
import pkg_resources

from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
from nose.plugins.attrib import attr

import shutil
import os

model_dir = "./temp"

learning_rate = 0.1
data_shape = (250, 250, 3)  # size of atnt images
data_type = tf.float32
batch_size = 16
validation_batch_size = 250
epochs = 1
steps = 5000

@attr('slow')
def test_logitstrainer_images():
    # Trainer logits
    try:
        embedding_validation = False
        trainer = Logits(
            model_dir=model_dir,
            architecture=dummy,
            optimizer=tf.train.GradientDescentOptimizer(learning_rate),
            n_classes=10,
            loss_op=mean_cross_entropy_loss,
            embedding_validation=embedding_validation,
            validation_batch_size=validation_batch_size,
            apply_moving_averages=False)
        run_logitstrainer_images(trainer)
    finally:
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass


def run_logitstrainer_images(trainer):
    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    filenames = [
        pkg_resources.resource_filename(
            __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
        pkg_resources.resource_filename(
            __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
        pkg_resources.resource_filename(
            __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
        pkg_resources.resource_filename(
            __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png')
    ]
    labels = [0, 0, 1, 1]

    def input_fn():

        return shuffle_data_and_labels_image_augmentation(
            filenames,
            labels,
            data_shape,
            data_type,
            batch_size,
            epochs=epochs)

    def input_fn_validation():
        return shuffle_data_and_labels_image_augmentation(
            filenames,
            labels,
            data_shape,
            data_type,
            validation_batch_size,
            epochs=1000)

    hooks = [
        LoggerHookEstimator(trainer, 16, 300),
        tf.train.SummarySaverHook(
            save_steps=1000,
            output_dir=model_dir,
            scaffold=tf.train.Scaffold(),
            summary_writer=tf.summary.FileWriter(model_dir))
    ]

    trainer.train(input_fn, steps=steps, hooks=hooks)

    acc = trainer.evaluate(input_fn_validation)
    assert acc['accuracy'] > 0.30, acc['accuracy']

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
