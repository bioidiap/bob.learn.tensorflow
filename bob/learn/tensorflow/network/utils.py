#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
slim = tf.contrib.slim


def append_logits(graph, n_classes, reuse):
    graph = slim.fully_connected(graph, n_classes, activation_fn=None, 
               weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
               weights_regularizer=slim.l2_regularizer(0.1),
               scope='Logits', reuse=reuse)

    return graph


