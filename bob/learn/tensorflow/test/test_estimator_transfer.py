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
from .test_estimator_onegraph import run_logitstrainer_mnist

import numpy

import shutil
import os


tfrecord_train = "./train_mnist.tfrecord"
tfrecord_validation = "./validation_mnist.tfrecord"    
model_dir = "./temp"
model_dir_adapted = "./temp2"

learning_rate = 0.1
data_shape = (28, 28, 1)  # size of atnt images
data_type = tf.float32
batch_size = 16
validation_batch_size = 250
epochs = 2
steps = 5000


def dummy_adapted(inputs, reuse=False, is_training_mode = True, trainable_variables=True):
    """
    Create all the necessary variables for this CNN

    **Parameters**
        inputs:
        
        reuse:
    """

    slim = tf.contrib.slim
    graph, end_points = dummy(inputs, reuse=reuse, is_training_mode = is_training_mode, trainable_variables=trainable_variables)

    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('Adapted', reuse=reuse):
        graph = slim.fully_connected(graph, 50,
                                     weights_initializer=initializer,
                                     activation_fn=tf.nn.relu,
                                     scope='fc2')
        end_points['fc2'] = graph

        graph = slim.fully_connected(graph, 25,
                                     weights_initializer=initializer,
                                     activation_fn=None,
                                     scope='fc3')
        end_points['fc3'] = graph


    return graph, end_points


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
        del trainer

        ## Again
        extra_checkpoint = {"checkpoint_path":"./temp", 
                            "scopes": dict({"Dummy/": "Dummy/"}),
                            "is_trainable": False
                           }

        trainer = Logits(model_dir=model_dir_adapted,
                                architecture=dummy_adapted,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                loss_op=mean_cross_entropy_loss,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size,
                                extra_checkpoint=extra_checkpoint
                                )
    
        run_logitstrainer_mnist(trainer, augmentation=True)

    finally:
        try:
            os.unlink(tfrecord_train)
            os.unlink(tfrecord_validation)
            shutil.rmtree(model_dir, ignore_errors=True)
            shutil.rmtree(model_dir_adapted, ignore_errors=True)
            pass
        except Exception:
            pass


def test_logitstrainer_center_loss():
    # Trainer logits
    try:        
        embedding_validation = False
        
        trainer = LogitsCenterLoss(model_dir=model_dir,
                                architecture=dummy,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size)
        run_logitstrainer_mnist(trainer, augmentation=True)
        del trainer

        ## Again
        extra_checkpoint = {"checkpoint_path":"./temp", 
                            "scopes": dict({"Dummy/": "Dummy/"}),
                            "is_trainable": False
                           }

        trainer = LogitsCenterLoss(model_dir=model_dir_adapted,
                                architecture=dummy_adapted,
                                optimizer=tf.train.GradientDescentOptimizer(learning_rate),
                                n_classes=10,
                                embedding_validation=embedding_validation,
                                validation_batch_size=validation_batch_size,
                                extra_checkpoint=extra_checkpoint
                                )
    
        run_logitstrainer_mnist(trainer, augmentation=True)

    finally:
        try:
            os.unlink(tfrecord_train)
            os.unlink(tfrecord_validation)
            shutil.rmtree(model_dir, ignore_errors=True)
            shutil.rmtree(model_dir_adapted, ignore_errors=True)
        except Exception:
            pass

