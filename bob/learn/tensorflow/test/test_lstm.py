#!/usr/bin/env python

import numpy
from bob.learn.tensorflow.datashuffler import Memory, ScaleFactor
from bob.learn.tensorflow.network import MLP, Embedding
from bob.learn.tensorflow.loss import BaseLoss
from bob.learn.tensorflow.trainers import Trainer, constant
from bob.learn.tensorflow.utils import load_mnist
import tensorflow as tf

train_data, train_labels, validation_data, validation_labels = load_mnist(data_dir="mnist")
