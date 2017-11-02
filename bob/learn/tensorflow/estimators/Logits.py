#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
import threading
import os
import bob.io.base
import bob.core
from ..analyzers import SoftmaxAnalizer
from tensorflow.core.framework import summary_pb2
import time

#logger = bob.core.log.setup("bob.learn.tensorflow")
from bob.learn.tensorflow.network.utils import append_logits
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from bob.learn.tensorflow.loss import mean_cross_entropy_center_loss

from . import check_features, is_trainable_checkpoint

import logging
logger = logging.getLogger("bob.learn")


class Logits(estimator.Estimator):
    """
    NN Trainer whose with logits as last layer

    The **architecture** function should follow the following pattern:

      def my_beautiful_function(placeholder):

          end_points = dict()
          graph = convXX(placeholder)
          end_points['conv'] = graph
          ....
          return graph, end_points

    The **loss** function should follow the following pattern:

    def my_beautiful_loss(logits, labels):
       return loss_set_of_ops(logits, labels)


    Variables, scopes... from other models can be loaded by the model_fn.
    For that, please, wrap the the path of the OTHER checkpoint and the list
    of variables in a dictionary with the key "load_variable_from_checkpoint" an provide them to the keyword `params`:
        
    {"load_variable_from_checkpoint": {"checkpoint_path":"mypath",
                                       "scopes":{"my_scope/": my_scope/}}}



    **Parameters**
      architecture:
         Pointer to a function that builds the graph.

      optimizer:
         One of the tensorflow solvers (https://www.tensorflow.org/api_guides/python/train)
         - tf.train.GradientDescentOptimizer
         - tf.train.AdagradOptimizer
         - ....
         
      config:
         
      n_classes:
         Number of classes of your problem. The logits will be appended in this class
         
      loss_op:
         Pointer to a function that computes the loss.
      
      embedding_validation:
         Run the validation using embeddings?? [default: False]
      
      model_dir:
        Model path

      validation_batch_size:
        Size of the batch for validation. This value is used when the
        validation with embeddings is used. This is a hack.
        
      params:
        Extra params for the model function (please see https://www.tensorflow.org/extend/estimators for more info)

      extra_checkpoint: dict()
        In case you want to use other model to initialize some variables.
        This argument should be in the following format
        extra_checkpoint = {"checkpoint_path": <YOUR_CHECKPOINT>, 
                            "scopes": dict({"<SOURCE_SCOPE>/": "<TARGET_SCOPE>/"}),
                            "is_trainable": <IF_THOSE_LOADED_VARIABLES_ARE_TRAINABLE>
                           }
    """

    def __init__(self,
                 architecture=None,
                 optimizer=None,
                 config=None,
                 n_classes=0,
                 loss_op=None,
                 embedding_validation=False,
                 model_dir="",
                 validation_batch_size=None,
                 params=None,
                 extra_checkpoint=None
              ):

        self.architecture = architecture
        self.optimizer=optimizer
        self.n_classes=n_classes
        self.loss_op=loss_op
        self.loss = None
        self.embedding_validation = embedding_validation
        self.extra_checkpoint = extra_checkpoint

        if self.architecture is None:
            raise ValueError("Please specify a function to build the architecture !!")
            
        if self.optimizer is None:
            raise ValueError("Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!")

        if self.loss_op is None:
            raise ValueError("Please specify a function to build the loss !!")

        if self.n_classes <= 0:
            raise ValueError("Number of classes must be greated than 0")

        def _model_fn(features, labels, mode, params, config):

            check_features(features)
            data = features['data']
            key = features['key']

            # Building one graph, by default everything is trainable
            if  self.extra_checkpoint is None:
                is_trainable = True
            else:
                is_trainable = is_trainable_checkpoint(self.extra_checkpoint)
            
            prelogits = self.architecture(data, is_trainable=is_trainable)[0]
            logits = append_logits(prelogits, n_classes)

            if self.embedding_validation:
                # Compute the embeddings
                embeddings = tf.nn.l2_normalize(prelogits, 1)
                predictions = {
                    "embeddings": embeddings
                }
            else:
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Compute Loss (for both TRAIN and EVAL modes)
            self.loss = self.loss_op(logits, labels)

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(self.extra_checkpoint["checkpoint_path"],
                                                              self.extra_checkpoint["scopes"])

                global_step = tf.contrib.framework.get_or_create_global_step()
                train_op = self.optimizer.minimize(self.loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss,
                                                  train_op=train_op)


            # Validation
            if self.embedding_validation:
                predictions_op = predict_using_tensors(predictions["embeddings"], labels, num=validation_batch_size)
                eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions_op)}
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)
            
            else:
                # Add evaluation metrics (for EVAL mode)
                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)

        super(Logits, self).__init__(model_fn=_model_fn,
                                     model_dir=model_dir,
                                     params=params,
                                     config=config)


class LogitsCenterLoss(estimator.Estimator):
    """
    NN Trainer whose with logits as last layer

    The **architecture** function should follow the following pattern:

      def my_beautiful_function(placeholder):

          end_points = dict()
          graph = convXX(placeholder)
          end_points['conv'] = graph
          ....
          return graph, end_points

    **Parameters**
      architecture:
         Pointer to a function that builds the graph.

      optimizer:
         One of the tensorflow solvers (https://www.tensorflow.org/api_guides/python/train)
         - tf.train.GradientDescentOptimizer
         - tf.train.AdagradOptimizer
         - ....

      config:

      n_classes:
         Number of classes of your problem. The logits will be appended in this class

      loss_op:
         Pointer to a function that computes the loss.

      embedding_validation:
         Run the validation using embeddings?? [default: False]

      model_dir:
        Model path

      validation_batch_size:
        Size of the batch for validation. This value is used when the
        validation with embeddings is used. This is a hack.
        
      params:
        Extra params for the model function (please see https://www.tensorflow.org/extend/estimators for more info)
        
    """

    def __init__(self,
                 architecture=None,
                 optimizer=None,
                 config=None,
                 n_classes=0,
                 embedding_validation=False,
                 model_dir="",
                 alpha=0.9,
                 factor=0.01,
                 validation_batch_size=None,
                 params=None,
                 extra_checkpoint=None,                 
              ):

        self.architecture = architecture
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.alpha = alpha
        self.factor = factor
        self.loss = None
        self.embedding_validation = embedding_validation
        self.extra_checkpoint = extra_checkpoint

        if self.architecture is None:
            raise ValueError("Please specify a function to build the architecture !!")

        if self.optimizer is None:
            raise ValueError("Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!")

        if self.n_classes <= 0:
            raise ValueError("Number of classes must be greated than 0")

        def _model_fn(features, labels, mode, params, config):

            check_features(features)
            data = features['data']
            key = features['key']

            # Building one graph, by default everything is trainable
            if  self.extra_checkpoint is None:
                is_trainable = True
            else:
                is_trainable = is_trainable_checkpoint(self.extra_checkpoint)
            
            prelogits = self.architecture(data)[0]
            logits = append_logits(prelogits, n_classes)

            # Compute Loss (for both TRAIN and EVAL modes)
            loss_dict = mean_cross_entropy_center_loss(logits, prelogits, labels, self.n_classes,
                                                       alpha=self.alpha, factor=self.factor)

            if self.embedding_validation:
                # Compute the embeddings
                embeddings = tf.nn.l2_normalize(prelogits, 1)
                predictions = {
                    "embeddings": embeddings
                }
            else:
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


            self.loss = loss_dict['loss']
            centers = loss_dict['centers']

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Loading variables from some model just in case
                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(self.extra_checkpoint["checkpoint_path"],
                                                              self.extra_checkpoint["scopes"])

                global_step = tf.contrib.framework.get_or_create_global_step()
                # backprop and updating the centers
                train_op = tf.group(self.optimizer.minimize(self.loss, global_step=global_step),
                                    centers)

                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss,
                                                  train_op=train_op)


            if self.embedding_validation:
                predictions_op = predict_using_tensors(predictions["embeddings"], labels, num=validation_batch_size)
                eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions_op)}
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)

            else:
                # Add evaluation metrics (for EVAL mode)
                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)

        super(LogitsCenterLoss, self).__init__(model_fn=_model_fn,
                                               model_dir=model_dir,
                                               config=config)
                                               