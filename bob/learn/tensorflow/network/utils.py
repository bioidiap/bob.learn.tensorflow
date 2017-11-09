#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf
import tensorflow.contrib.slim as slim


def append_logits(graph, n_classes, reuse=False, l2_regularizer=0.001,
                  weights_std=0.1):
    return slim.fully_connected(
        graph, n_classes, activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=weights_std),
        weights_regularizer=slim.l2_regularizer(l2_regularizer),
        scope='Logits', reuse=reuse)
