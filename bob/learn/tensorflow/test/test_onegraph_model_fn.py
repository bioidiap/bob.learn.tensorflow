#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.trainers import LogitsTrainer, LogitsCenterLossTrainer
from bob.learn.tensorflow.utils.tfrecords import shuffle_data_and_labels, batch_data_and_labels
from bob.learn.tensorflow.utils import load_mnist, create_mnist_tfrecord
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
import numpy

import shutil
import os


tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"    
model_dir = "./temp"

learning_rate = 0.1
data_shape = (28, 28, 1)  # size of atnt images
data_type = tf.float32
batch_size = 16
validation_batch_size = 1000
epochs = 1
steps = 2000


def test_logitstrainer():
    run_logitstrainer(False)


def test_logitstrainer_embedding():
    run_logitstrainer(True)


def test_logitstrainer_centerloss():
    run_logitstrainer_centerloss(False)


def test_logitstrainer_centerloss_embedding():
    run_logitstrainer_centerloss(True)


def run_logitstrainer(embedding_validation):

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Creating tf records for mnist
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    create_mnist_tfrecord(tfrecord_train, train_data, train_labels, n_samples=6000)
    create_mnist_tfrecord(tfrecord_validation, validation_data, validation_labels, n_samples=1000)

    try:
        
        # Trainer logits
        trainer = LogitsTrainer(model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                loss_op=mean_cross_entropy_loss,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size
                                )

        def input_fn():
            return shuffle_data_and_labels(tfrecord_train, data_shape, data_type,
                                           batch_size, epochs=epochs)
                                       
        def input_fn_validation():
            return batch_data_and_labels(tfrecord_validation, data_shape, data_type,
                                         validation_batch_size, epochs=epochs)                                       

        hooks = [LoggerHookEstimator(trainer, 16, 100)]
        trainer.train(input_fn, steps=steps, hooks=hooks)
        
        if not embedding_validation:

            acc = trainer.evaluate(input_fn_validation)
            assert acc['accuracy'] > 0.80
        else:
            acc = trainer.evaluate(input_fn_validation)
            assert acc['accuracy'] > 0.80

    finally:
        try:
            os.unlink(tfrecord_train)
            os.unlink(tfrecord_validation)            
            shutil.rmtree(model_dir)
        except Exception:
            pass

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


def run_logitstrainer_centerloss(embedding_validation):

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Creating tf records for mnist
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    create_mnist_tfrecord(tfrecord_train, train_data, train_labels, n_samples=6000)
    create_mnist_tfrecord(tfrecord_validation, validation_data, validation_labels, n_samples=1000)

    try:

        # Trainer logits
        trainer = LogitsCenterLossTrainer(
                                model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size,
                                factor=0.01
                                )

        def input_fn():
            return shuffle_data_and_labels(tfrecord_train, data_shape, data_type,
                                           batch_size, epochs=epochs)

        def input_fn_validation():
            return batch_data_and_labels(tfrecord_validation, data_shape, data_type,
                                         validation_batch_size, epochs=epochs)

        hooks = [LoggerHookEstimator(trainer, 16, 100)]
        trainer.train(input_fn, steps=steps, hooks=hooks)

        if not embedding_validation:
            acc = trainer.evaluate(input_fn_validation)
            assert acc['accuracy'] > 0.80
        else:
            acc = trainer.evaluate(input_fn_validation)
            assert acc['accuracy'] > 0.80

        sess = tf.Session()
        checkpoint_path = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta", clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        centers = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="center_loss/centers:0")[0]
        assert numpy.sum(numpy.abs(centers.eval(sess))) > 0.0

    finally:
        try:
            os.unlink(tfrecord_train)
            os.unlink(tfrecord_validation)
            shutil.rmtree(model_dir)
        except Exception:
            pass

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
