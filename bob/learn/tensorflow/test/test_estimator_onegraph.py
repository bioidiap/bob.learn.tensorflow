#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from nose.plugins.attrib import attr
import tensorflow as tf

from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.estimators import Logits, LogitsCenterLoss

from bob.learn.tensorflow.dataset.tfrecords import shuffle_data_and_labels, batch_data_and_labels, \
    shuffle_data_and_labels_image_augmentation

from bob.learn.tensorflow.utils import load_mnist, create_mnist_tfrecord
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
from bob.learn.tensorflow.utils import reproducible

import numpy

import shutil
import os

# Fixing problem with MAC https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"
model_dir = "./temp"

learning_rate = 0.1
data_shape = (28, 28, 1)  # size of atnt images
data_type = tf.float32
batch_size = 32
validation_batch_size = 250
epochs = 6
steps = 5000
reproducible.set_seed()

# @attr('slow')
# def test_logitstrainer():
#     # Trainer logits
#     try:
#         embedding_validation = False
#         _, run_config, _, _, _ = reproducible.set_seed()
#         run_config = run_config.replace(
#             keep_checkpoint_max=10, save_checkpoints_steps=100, save_checkpoints_secs=None)
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
#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#         except Exception:
#             pass

# @attr('slow')
# def test_logitstrainer_embedding():
#     try:
#         embedding_validation = True
#         _, run_config, _, _, _ = reproducible.set_seed()
#         trainer = Logits(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             config=run_config)

#         run_logitstrainer_mnist(trainer)
#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#         except Exception:
#             pass

# @attr('slow')
# def test_logitstrainer_centerloss():
#     try:
#         embedding_validation = False
#         _, run_config, _, _, _ = reproducible.set_seed()
#         run_config = run_config.replace(save_checkpoints_steps=1000)
#         trainer = LogitsCenterLoss(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             factor=0.01,
#             config=run_config)

#         run_logitstrainer_mnist(trainer)

#         # Checking if the centers were updated
#         sess = tf.Session()
#         checkpoint_path = tf.train.get_checkpoint_state(
#             model_dir).model_checkpoint_path
#         saver = tf.train.import_meta_graph(
#             checkpoint_path + ".meta", clear_devices=True)
#         saver.restore(sess, tf.train.latest_checkpoint(model_dir))
#         centers = tf.get_collection(
#             tf.GraphKeys.GLOBAL_VARIABLES, scope="center_loss/centers:0")[0]
#         assert numpy.sum(numpy.abs(centers.eval(sess))) > 0.0

#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#         except Exception:
#             pass

# @attr('slow')
# def test_logitstrainer_centerloss_embedding():
#     try:
#         embedding_validation = True
#         _, run_config, _, _, _ = reproducible.set_seed()
#         trainer = LogitsCenterLoss(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             embedding_validation=embedding_validation,
#             validation_batch_size=validation_batch_size,
#             factor=0.01,
#             config=run_config)
#         run_logitstrainer_mnist(trainer)

#         # Checking if the centers were updated
#         sess = tf.Session()
#         checkpoint_path = tf.train.get_checkpoint_state(
#             model_dir).model_checkpoint_path
#         saver = tf.train.import_meta_graph(
#             checkpoint_path + ".meta", clear_devices=True)
#         saver.restore(sess, tf.train.latest_checkpoint(model_dir))
#         centers = tf.get_collection(
#             tf.GraphKeys.GLOBAL_VARIABLES, scope="center_loss/centers:0")[0]
#         assert numpy.sum(numpy.abs(centers.eval(sess))) > 0.0
#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#         except Exception:
#             pass


def run_logitstrainer_mnist(trainer, augmentation=False):
    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

    # Creating tf records for mnist
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    create_mnist_tfrecord(
        tfrecord_train, train_data, train_labels, n_samples=6000)
    create_mnist_tfrecord(
        tfrecord_validation,
        validation_data,
        validation_labels,
        n_samples=validation_batch_size)

    def input_fn():
        if augmentation:
            return shuffle_data_and_labels_image_augmentation(
                tfrecord_train,
                data_shape,
                data_type,
                batch_size,
                random_flip=True,
                random_rotate=False,
                epochs=epochs)
        else:
            return shuffle_data_and_labels(
                tfrecord_train,
                data_shape,
                data_type,
                batch_size,
                epochs=epochs)

    def input_fn_validation():
        return batch_data_and_labels(
            tfrecord_validation,
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
    if not trainer.embedding_validation:
        acc = trainer.evaluate(input_fn_validation)
        assert acc['accuracy'] > 0.10
    else:
        acc = trainer.evaluate(input_fn_validation)
        assert acc['accuracy'] > 0.10

    # Cleaning up
    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0

# @attr('slow')
# def test_moving_average_trainer():
#     # define a fixed input data
#     # train the same network with the same initialization
#     # evaluate it
#     # train and evaluate it again with moving average
#     # Accuracy should be lower when moving average is on

#     try:
#         # Creating tf records for mnist
#         train_data, train_labels, validation_data, validation_labels = load_mnist()
#         create_mnist_tfrecord(
#             tfrecord_train, train_data, train_labels, n_samples=6000)
#         create_mnist_tfrecord(
#             tfrecord_validation,
#             validation_data,
#             validation_labels,
#             n_samples=validation_batch_size)

#         def input_fn():
#             return batch_data_and_labels(
#                 tfrecord_train,
#                 data_shape,
#                 data_type,
#                 batch_size,
#                 epochs=1)

#         def input_fn_validation():
#             return batch_data_and_labels(
#                 tfrecord_validation,
#                 data_shape,
#                 data_type,
#                 validation_batch_size,
#                 epochs=1)

#         from bob.learn.tensorflow.network.Dummy import dummy as architecture

#         run_config = reproducible.set_seed(183, 183)[1]
#         run_config = run_config.replace(save_checkpoints_steps=2000)

#         def _estimator(apply_moving_averages):
#             return Logits(
#                 architecture,
#                 tf.train.GradientDescentOptimizer(1e-1),
#                 tf.losses.sparse_softmax_cross_entropy,
#                 10,
#                 model_dir=model_dir,
#                 config=run_config,
#                 apply_moving_averages=apply_moving_averages,
#             )

#         def _evaluate(estimator, delete=True):
#             try:
#                 estimator.train(input_fn)
#                 evaluations = estimator.evaluate(input_fn_validation)
#             finally:
#                 if delete:
#                     shutil.rmtree(estimator.model_dir, ignore_errors=True)
#             return evaluations

#         estimator = _estimator(False)
#         evaluations = _evaluate(estimator, delete=True)
#         no_moving_average_acc = evaluations['accuracy']

#         # same as above with moving average
#         estimator = _estimator(True)
#         evaluations = _evaluate(estimator, delete=False)
#         with_moving_average_acc = evaluations['accuracy']

#         assert no_moving_average_acc > with_moving_average_acc, \
#             (no_moving_average_acc, with_moving_average_acc)

#         # Can it resume training?
#         del estimator
#         tf.reset_default_graph()
#         estimator = _estimator(True)
#         _evaluate(estimator, delete=True)

#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#         except Exception:
#             pass

# @attr('slow')
# def test_saver_with_moving_average():
#     try:
#         _, run_config, _, _, _ = reproducible.set_seed()
#         run_config = run_config.replace(
#             keep_checkpoint_max=10, save_checkpoints_steps=100,
#             save_checkpoints_secs=None)
#         estimator = Logits(
#             model_dir=model_dir,
#             architecture=dummy,
#             optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#             n_classes=10,
#             loss_op=mean_cross_entropy_loss,
#             embedding_validation=False,
#             validation_batch_size=validation_batch_size,
#             config=run_config)
#         run_logitstrainer_mnist(estimator, augmentation=True)
#         ckpt = tf.train.get_checkpoint_state(estimator.model_dir)
#         assert ckpt, "Failed to get any checkpoint!"
#         assert len(
#             ckpt.all_model_checkpoint_paths) == 10, ckpt.all_model_checkpoint_paths
#     finally:
#         try:
#             os.unlink(tfrecord_train)
#             os.unlink(tfrecord_validation)
#             shutil.rmtree(model_dir, ignore_errors=True)
#         except Exception:
#             pass
