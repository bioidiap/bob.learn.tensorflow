#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains MNIST with LENET using Tensor flow

Usage:
  train_mnist_triplet.py [--batch-size=<arg> --validation-batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_mnist_triplet.py -h | --help
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
from bob.learn.tensorflow.data import MemoryDataShuffler, TextDataShuffler
from bob.learn.tensorflow.network import Lenet, MLP, LenetDropout, VGG, Chopra, Dummy
from bob.learn.tensorflow.trainers import TripletTrainer
from bob.learn.tensorflow.loss import TripletLoss
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

    train_data_shuffler = MemoryDataShuffler(train_data, train_labels,
                                             input_shape=[28, 28, 1],
                                             scale=True,
                                             batch_size=BATCH_SIZE)

    validation_data_shuffler = MemoryDataShuffler(validation_data, validation_labels,
                                                  input_shape=[28, 28, 1],
                                                  scale=True,
                                                  batch_size=VALIDATION_BATCH_SIZE)

    # Preparing the architecture
    n_classes = len(train_data_shuffler.possible_labels)
    #n_classes = 200
    cnn = True
    if cnn:
        architecture = Chopra(seed=SEED, fc1_output=n_classes)
        #architecture = Lenet(default_feature_layer="fc2", n_classes=n_classes, conv1_output=8, conv2_output=16,use_gpu=USE_GPU)
        #architecture = VGG(n_classes=n_classes, use_gpu=USE_GPU)
        #architecture = Dummy(seed=SEED)
        #architecture = LenetDropout(default_feature_layer="fc2", n_classes=n_classes, conv1_output=4, conv2_output=8, use_gpu=USE_GPU)

        loss = TripletLoss()
        optimizer = tf.train.GradientDescentOptimizer(0.000001)
        trainer = TripletTrainer(architecture=architecture,
                                 loss=loss,
                                 iterations=ITERATIONS,
                                 snapshot=VALIDATION_TEST,
                                 temp_dir="cnn-triplet",
                                 prefetch=True,
                                 optimizer=optimizer
                                 )
        trainer.train(train_data_shuffler, validation_data_shuffler)
        #trainer.train(train_data_shuffler)
    else:
        mlp = MLP(n_classes, hidden_layers=[15, 20])

        loss = TripletLoss()
        trainer = TripletTrainer(architecture=mlp,
                                 loss=loss,
                                 temp_dir="dnn-triplet",
                                 iterations=ITERATIONS,
                                 snapshot=VALIDATION_TEST)
        trainer.train(train_data_shuffler, validation_data_shuffler)

