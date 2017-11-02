#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Logits, LogitsCenterLoss

from bob.learn.tensorflow.dataset.tfrecords import shuffle_data_and_labels, batch_data_and_labels, shuffle_data_and_labels_image_augmentation


from bob.learn.tensorflow.dataset import append_image_augmentation
from bob.learn.tensorflow.utils import load_mnist, create_mnist_tfrecord
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.utils import reproducible
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
validation_batch_size = 250
epochs = 1
steps = 5000


def test_logitstrainer():
    # Trainer logits
    try:
        embedding_validation = False
        trainer = Logits(model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                loss_op=mean_cross_entropy_loss,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size)
        run_logitstrainer_mnist(trainer, augmentation=True)
    finally:
        try:
            os.unlink(tfrecord_train)
            os.unlink(tfrecord_validation)
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass        


def test_logitstrainer_embedding():
    try:
        embedding_validation = True
        trainer = Logits(model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                loss_op=mean_cross_entropy_loss,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size)

        run_logitstrainer_mnist(trainer)
    finally:
        try:
            os.unlink(tfrecord_train)
            os.unlink(tfrecord_validation)
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass        


def test_logitstrainer_centerloss():

    try:
        embedding_validation = False
        run_config = tf.estimator.RunConfig()
        run_config = run_config.replace(save_checkpoints_steps=1000)
        trainer = LogitsCenterLoss(
                                model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size,
                                factor=0.01,
                                config=run_config)
                                
        run_logitstrainer_mnist(trainer)

        # Checking if the centers were updated
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
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass


def test_logitstrainer_centerloss_embedding():
    try:
        embedding_validation = True
        trainer = LogitsCenterLoss(
                                model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size,
                                factor=0.01)
        run_logitstrainer_mnist(trainer)
        
        # Checking if the centers were updated
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
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception:
            pass        


def run_logitstrainer_mnist(trainer, augmentation=False):

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Creating tf records for mnist
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    create_mnist_tfrecord(tfrecord_train, train_data, train_labels, n_samples=6000)
    create_mnist_tfrecord(tfrecord_validation, validation_data, validation_labels, n_samples=validation_batch_size)

    def input_fn():
        if augmentation:
            return shuffle_data_and_labels_image_augmentation(tfrecord_train, data_shape, data_type, batch_size, epochs=epochs)
        else:
            return shuffle_data_and_labels(tfrecord_train, data_shape, data_type,
                                           batch_size, epochs=epochs)
        

    def input_fn_validation():
        return batch_data_and_labels(tfrecord_validation, data_shape, data_type,
                                     validation_batch_size, epochs=1000)
    
    hooks = [LoggerHookEstimator(trainer, 16, 300),

             tf.train.SummarySaverHook(save_steps=1000,
                                       output_dir=model_dir,
                                       scaffold=tf.train.Scaffold(),
                                       summary_writer=tf.summary.FileWriter(model_dir) )]

    trainer.train(input_fn, steps=steps, hooks=hooks)

    if not trainer.embedding_validation:
        acc = trainer.evaluate(input_fn_validation)
        assert acc['accuracy'] > 0.40
    else:
        acc = trainer.evaluate(input_fn_validation)
        assert acc['accuracy'] > 0.40

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
