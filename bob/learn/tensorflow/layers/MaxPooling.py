#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 17:38 CEST

import tensoflow as tf
from bob.learn.tensorflow.util import *
from .Layer import Layer


class MaxPooling(Layer):

    