#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
import os
import bob.io.base
import bob.core
from tensorflow.core.framework import summary_pb2
import time

#logger = bob.core.log.setup("bob.learn.tensorflow")
from tensorflow.python.estimator import estimator
from bob.learn.tensorflow.utils import predict_using_tensors
#from bob.learn.tensorflow.loss import mean_cross_entropy_center_loss
from . import check_features, is_trainable_checkpoint

import logging
logger = logging.getLogger("bob.learn")


from bob.learn.tensorflow.network.utils import append_logits
from bob.learn.tensorflow.loss import mean_cross_entropy_loss


class Siamese(estimator.Estimator):
    """
    NN estimator for Siamese networks

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


        extra_checkpoint = {"checkpoint_path":model_dir, 
                            "scopes": dict({"Dummy/": "Dummy/"}),
                            "is_trainable": False
                           }




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
                 loss_op=None,
                 model_dir="",
                 validation_batch_size=None,
                 params=None,
                 extra_checkpoint=None                 
              ):

        self.architecture = architecture
        self.optimizer=optimizer
        self.loss_op=loss_op
        self.loss = None
        self.extra_checkpoint = extra_checkpoint        

        if self.architecture is None:
            raise ValueError("Please specify a function to build the architecture !!")
            
        if self.optimizer is None:
            raise ValueError("Please specify a optimizer (https://www.tensorflow.org/api_guides/python/train) !!")

        if self.loss_op is None:
            raise ValueError("Please specify a function to build the loss !!")

        def _model_fn(features, labels, mode, params, config):

            if mode == tf.estimator.ModeKeys.TRAIN:
                # Building one graph, by default everything is trainable
                if  self.extra_checkpoint is None:
                    is_trainable = True
                else:
                    is_trainable = is_trainable_checkpoint(self.extra_checkpoint)

                # The input function needs to have dictionary pair with the `left` and `right` keys
                if not 'left' in features.keys() or not 'right' in features.keys():
                    raise ValueError("The input function needs to contain a dictionary with the keys `left` and `right` ")

                # Building one graph
                prelogits_left = self.architecture(features['left'], is_trainable=is_trainable)[0]
                prelogits_right = self.architecture(features['right'], reuse=True, is_trainable=is_trainable)[0]

                if self.extra_checkpoint is not None:
                    tf.contrib.framework.init_from_checkpoint(self.extra_checkpoint["checkpoint_path"],
                                                              self.extra_checkpoint["scopes"])
    
                # Compute Loss (for both TRAIN and EVAL modes)
                self.loss = self.loss_op(prelogits_left, prelogits_left, labels)
               
                # Configure the Training Op (for TRAIN mode)
                global_step = tf.contrib.framework.get_or_create_global_step()
                train_op = self.optimizer.minimize(self.loss, global_step=global_step)

                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss,
                                                  train_op=train_op)

            check_features(features)
            data = features['data']

            # Compute the embeddings
            prelogits = self.architecture(data)[0]
            embeddings = tf.nn.l2_normalize(prelogits, 1)
            predictions = {"embeddings": embeddings}

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            predictions_op = predict_using_tensors(predictions["embeddings"], labels, num=validation_batch_size)
            eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions_op)}
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=tf.reduce_mean(1), eval_metric_ops=eval_metric_ops)
            

        super(Siamese, self).__init__(model_fn=_model_fn,
                                     model_dir=model_dir,
                                     params=params,
                                     config=config)

