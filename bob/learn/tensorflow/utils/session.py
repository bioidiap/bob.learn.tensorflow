#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import tensorflow as tf
from .singleton import Singleton


@Singleton
class Session(object):

    def __init__(self):
        config = tf.ConfigProto(log_device_placement=True,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))
        config.gpu_options.allow_growth = True
        self.session = tf.Session()
