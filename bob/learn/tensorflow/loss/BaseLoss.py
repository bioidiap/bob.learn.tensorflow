#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 16:38 CEST

import logging
import tensorflow as tf
logger = logging.getLogger("bob.learn.tensorflow")

slim = tf.contrib.slim


class BaseLoss(object):
    """
    Base loss function.

    One exam
    """

    def __init__(self, loss, operation, name="loss"):
        self.loss = loss
        self.operation = operation
        self.name = name

    def __call__(self, graph, label):
        return self.operation(self.loss(logits=graph, labels=label), name=self.name)
        
        
class MeanSoftMaxLoss(object):
    """
    Mean softmax loss. Basically it wrapps the function tf.nn.sparse_softmax_cross_entropy_with_logits.
    """

    def __init__(self, name="loss", add_regularization_losses=True):
        """
        Constructor
        
        **Parameters**

          name:
             Scope name
        
        """
    
        self.name = name
        self.add_regularization_losses = add_regularization_losses

    def __call__(self, graph, label):
    
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                          logits=graph, labels=label), name=self.name)
    
        if self.add_regularization_losses:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return tf.add_n([loss] + regularization_losses, name='total_loss')
        else:
            return loss
            
class MeanSoftMaxLossCenterLoss(object):
    """
    Mean softmax loss. Basically it wrapps the function tf.nn.sparse_softmax_cross_entropy_with_logits.
    """

    def __init__(self, name="loss", add_regularization_losses=True, alpha=0.9, factor=0.01, n_classes=10):
        """
        Constructor
        
        **Parameters**

          name:
             Scope name
        
        """
    
        self.name = name
        self.add_regularization_losses = add_regularization_losses

        self.n_classes = n_classes
        self.alpha = alpha
        self.factor = factor


    def append_center_loss(self, features, label):
        nrof_features = features.get_shape()[1]
        
        centers = tf.get_variable('centers', [self.n_classes, nrof_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
            
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label)
        diff = (1 - self.alpha) * (centers_batch - features)
        centers = tf.scatter_sub(centers, label, diff)
        loss = tf.reduce_mean(tf.square(features - centers_batch))
        
        return loss


    def __call__(self, logits_prelogits, label):
    
        #TODO: Test the dictionary
    
        logits = logits_prelogits['logits']
    
        # Cross entropy
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                          logits=logits, labels=label), name=self.name)

        # Appending center loss
        prelogits = logits_prelogits['prelogits']
        center_loss = self.append_center_loss(prelogits, label)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, center_loss * self.factor)
    
        # Adding the regularizers in the loss
        if self.add_regularization_losses:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss =  tf.add_n([loss] + regularization_losses, name='total_loss')
            
        return loss            
