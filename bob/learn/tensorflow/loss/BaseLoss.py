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

    def __init__(self, name="loss"):
        """
        Constructor
        
        **Parameters**

          name:
             Scope name
        
        """
    
        self.name = name

    def __call__(self, graph, label):
    
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                              logits=graph, labels=label), name=self.name)
