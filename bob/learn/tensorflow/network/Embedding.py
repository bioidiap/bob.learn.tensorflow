#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


import tensorflow as tf
from bob.learn.tensorflow.utils.session import Session
from bob.learn.tensorflow.datashuffler import Linear


class Embedding(object):
    """
    Embedding abstraction
    
    **Parameters**
    
      input: Input placeholder
      
      graph: Embedding graph
    
    """
    def __init__(self, input, graph, normalizer=Linear()):
        self.input = input
        self.graph = graph
        self.normalizer = normalizer

    def __call__(self, data):
        session = Session.instance().session

        if self.normalizer is not None:
            for i in range(data.shape[0]):
                data[i] = self.normalizer(data[i])

        feed_dict = {self.input: data}

        return session.run([self.graph], feed_dict=feed_dict)[0]
