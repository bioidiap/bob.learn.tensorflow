#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Siamese
from bob.learn.tensorflow.dataset.siamese_image import shuffle_data_and_labels_image_augmentation as siamese_batch
from bob.learn.tensorflow.dataset.image import shuffle_data_and_labels_image_augmentation as single_batch

from bob.learn.tensorflow.loss import contrastive_loss
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.utils import reproducible
import pkg_resources

import numpy
import shutil
import os


tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"    
model_dir = "./temp"

learning_rate = 0.001
data_shape = (250, 250, 3)  # size of atnt images
output_shape = (50, 50)
data_type = tf.float32
batch_size = 4
validation_batch_size = 2
epochs = 1
steps = 5000


# Data
filenames = [pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),                 
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),                 

                              
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),                 
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),                 
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),                 
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),                 
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
             pkg_resources.resource_filename(__name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
             ]
labels = [0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_logitstrainer():
    # Trainer logits
    try:
        trainer = Siamese(model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                loss_op=contrastive_loss,
                                validation_batch_size=validation_batch_size)
        run_siamesetrainer(trainer)
    finally:
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
            #pass
        except Exception:
            pass        


def run_siamesetrainer(trainer):

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    def input_fn():
        return siamese_batch(filenames, labels, data_shape, data_type, batch_size, epochs=epochs, output_shape=output_shape,
                             random_flip=True, random_brightness=True, random_contrast=True, random_saturation=True)

    def input_validation_fn():
        return single_batch(filenames, labels, data_shape, data_type, validation_batch_size, epochs=10, output_shape=output_shape)

    hooks = [LoggerHookEstimator(trainer, batch_size, 300),

             tf.train.SummarySaverHook(save_steps=1000,
                                       output_dir=model_dir,
                                       scaffold=tf.train.Scaffold(),
                                       summary_writer=tf.summary.FileWriter(model_dir) )]

    trainer.train(input_fn, steps=steps, hooks=hooks)

    acc = trainer.evaluate(input_validation_fn)
    assert acc['accuracy'] > 0.5

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

