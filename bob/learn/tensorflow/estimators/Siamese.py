#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
from . import check_features, get_trainable_variables

import logging

logger = logging.getLogger("bob.learn")


class Siamese(estimator.Estimator):
    """
    NN estimator for Siamese Networks proposed in: "Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity metric discriminatively, with application to face verification." Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005."

    The **architecture** function should follow the following pattern::

      ```
      def my_beautiful_function(placeholder):

          end_points = dict()
          graph = convXX(placeholder)
          end_points['conv'] = graph

          return graph, end_points
      ```


    The **loss** function should follow the following pattern::

      ```
      def my_beautiful_loss(logits, labels):
        return loss_set_of_ops(logits, labels)


        extra_checkpoint = {"checkpoint_path":model_dir,
                            "scopes": dict({"Dummy/": "Dummy/"}),
                            "trainable_variables": [<LIST OF VARIABLES OR SCOPES THAT YOU WANT TO TRAIN>]
                           }
      ```

    **Parameters**
      architecture:
         Pointer to a function that builds the graph.

      optimizer:
         One of the tensorflow solvers (https://www.tensorflow.org/api_guides/python/train)
         - tf.train.GradientDescentOptimizer
         - tf.train.AdagradOptimizer
         - ....

      config:

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
        Extra params for the model function
        (please see https://www.tensorflow.org/extend/estimators for more info)
        
      extra_checkpoint: dict
        In case you want to use other model to initialize some variables.
 
        This argument should be in the following format::
        ```
        extra_checkpoint = {"checkpoint_path": <YOUR_CHECKPOINT>,\
                            "scopes": dict({"<SOURCE_SCOPE>/": "<TARGET_SCOPE>/"}),\
                            "trainable_variables": [<LIST OF VARIABLES OR SCOPES THAT YOU WANT TO RETRAIN>]\
                            }
        ```
    """

    def __init__(self,
                 architecture=None,
                 optimizer=None,
                 config=None,
                 loss_op=None,
                 model_dir="",
                 validation_batch_size=None,
                 params=None,
                 extra_checkpoint=None):

        self.architecture = architecture
        self.optimizer = optimizer
        self.loss_op = loss_op
        self.loss = None
        self.extra_checkpoint = extra_checkpoint

        if self.architecture is None:
            raise ValueError(
                "Please specify a function to build the architecture !!")

        if self.optimizer is None:
            raise ValueError(
                "Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!"
            )

        if self.loss_op is None:
            raise ValueError("Please specify a function to build the loss !!")

        def _model_fn(features, labels, mode, params, config):
            if mode == tf.estimator.ModeKeys.TRAIN:

                # Building one graph, by default everything is trainable
                # The input function needs to have dictionary pair with the `left` and `right` keys
                if 'left' not in features.keys(
                ) or 'right' not in features.keys():
                    raise ValueError(
                        "The input function needs to contain a dictionary with the keys `left` and `right` "
                    )

                # Building one graph
                trainable_variables = get_trainable_variables(
                    self.extra_checkpoint)
                prelogits_left, end_points_left = self.architecture(
                    features['left'],
                    mode=mode,
                    trainable_variables=trainable_variables)
                prelogits_right, end_points_right = self.architecture(
                    features['right'],
                    reuse=True,
                    mode=mode,
                    trainable_variables=trainable_variables)

                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(
                        self.extra_checkpoint["checkpoint_path"],
                        self.extra_checkpoint["scopes"])

                # Compute Loss (for both TRAIN and EVAL modes)
                self.loss = self.loss_op(prelogits_left, prelogits_right,
                                         labels)

                # Configure the Training Op (for TRAIN mode)
                global_step = tf.train.get_or_create_global_step()
                train_op = self.optimizer.minimize(
                    self.loss, global_step=global_step)

                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=self.loss, train_op=train_op)

            check_features(features)
            data = features['data']

            # Compute the embeddings
            prelogits = self.architecture(data, mode=mode)[0]
            embeddings = tf.nn.l2_normalize(prelogits, 1)
            predictions = {"embeddings": embeddings}

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)

            predictions_op = predict_using_tensors(
                predictions["embeddings"], labels, num=validation_batch_size)
            eval_metric_ops = {
                "accuracy":
                tf.metrics.accuracy(labels=labels, predictions=predictions_op)
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.reduce_mean(1),
                eval_metric_ops=eval_metric_ops)

        super(Siamese, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            params=params,
            config=config)
