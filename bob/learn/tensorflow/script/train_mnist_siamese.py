#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains MNIST with LENET using Tensor flow

Usage:
  train_mnist_siamese.py [--batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_mnist_siamese.py -h | --help
Options:
  -h --help     Show this screen.
  --batch-size=<arg>  [default: 1]
  --iterations=<arg>  [default: 30000]
  --validation-interval=<arg>  [default: 100]
"""

from docopt import docopt
import tensorflow as tf
from .. import util
SEED = 10
from bob.learn.tensorflow.data import PairDataShuffler
from bob.learn.tensorflow.network import Lenet
from bob.learn.tensorflow.trainers import SiameseTrainer
from bob.learn.tensorflow.loss import ContrastiveLoss

import numpy

def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    USE_GPU = args['--use-gpu']
    perc_train = 0.9

    # Loading data
    data, labels = util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
    data = numpy.reshape(data, (data.shape[0], 28, 28, 1))

    data_shuffler = PairDataShuffler(data, labels)

    # Preparing the architecture
    lenet = Lenet(feature_layer="fc2")

    loss = ContrastiveLoss()
    trainer = SiameseTrainer(architecture=lenet, loss=loss, iterations=ITERATIONS, base_lr=0.00001)
    trainer.train(data_shuffler)


