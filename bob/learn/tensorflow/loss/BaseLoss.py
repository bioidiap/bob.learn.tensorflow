#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Tue 09 Aug 2016 16:38 CEST

import logging
logger = logging.getLogger("bob.learn.tensorflow")


class BaseLoss(object):
    """
    Base loss function.

    One exam
    """

    def __init__(self, loss, operation):
        self.loss = loss
        self.operation = operation

    def __call__(self, graph, label):
        return self.operation(self.loss(graph, label))
