#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains MNIST with LENET using Tensor flow

Usage:
  train_mnist_siamese.py [--batch-size=<arg> --validation-batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_mnist_siamese.py -h | --help
Options:
  -h --help     Show this screen.
  --batch-size=<arg>  [default: 1]
  --validation-batch-size=<arg>   [default:128]
  --iterations=<arg>  [default: 30000]
  --validation-interval=<arg>  [default: 100]
"""

from docopt import docopt
import tensorflow as tf
from .. import util
SEED = 10
from bob.learn.tensorflow.datashuffler import SiameseMemory
from bob.learn.tensorflow.network import Lenet, MLP, LenetDropout, VGG, Chopra, Dummy
from bob.learn.tensorflow.trainers import SiameseTrainer
from bob.learn.tensorflow.loss import ContrastiveLoss
import numpy


def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    VALIDATION_BATCH_SIZE = int(args['--validation-batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    USE_GPU = args['--use-gpu']
    perc_train = 0.9

    # Loading data

    train_data, train_labels, validation_data, validation_labels = \
        util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))

    train_data_shuffler = SiameseMemory(train_data, train_labels,
                                        input_shape=[28, 28, 1],
                                        scale=True,
                                        batch_size=BATCH_SIZE)

    validation_data_shuffler = SiameseMemory(validation_data, validation_labels,
                                             input_shape=[28, 28, 1],
                                             scale=True,
                                             batch_size=VALIDATION_BATCH_SIZE)

    # Preparing the architecture
    n_classes = len(train_data_shuffler.possible_labels)
    cnn = True
    if cnn:

        # LENET PAPER CHOPRA
        architecture = Chopra(seed=SEED, fc1_output=n_classes)

        loss = ContrastiveLoss(contrastive_margin=4.)
        #optimizer = tf.train.GradientDescentOptimizer(0.000001)
        trainer = SiameseTrainer(architecture=architecture,
                                 loss=loss,
                                 iterations=ITERATIONS,
                                 snapshot=VALIDATION_TEST,
                                 prefetch=False,
                                 temp_dir="./LOGS/siamese-cnn-prefetch")

        trainer.train(train_data_shuffler, validation_data_shuffler)
    else:
        mlp = MLP(n_classes, hidden_layers=[15, 20])

        loss = ContrastiveLoss()
        trainer = SiameseTrainer(architecture=mlp,
                                 loss=loss,
                                 iterations=ITERATIONS,
                                 snapshot=VALIDATION_TEST,
                                 temp_dir="./LOGS/siamese-dnn")
        trainer.train(train_data_shuffler, validation_data_shuffler)
