#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains MNIST with LENET using Tensor flow

Usage:
  train_mnist.py [--batch-size=<arg> --validation-batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_mnist.py -h | --help
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
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory
from bob.learn.tensorflow.network import Lenet, MLP, Dummy, Chopra
from bob.learn.tensorflow.trainers import Trainer
from bob.learn.tensorflow.loss import BaseLoss
import bob.io.base
from ..analyzers import ExperimentAnalizer, SoftmaxAnalizer

import numpy

def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    VALIDATION_BATCH_SIZE = int(args['--validation-batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    USE_GPU = args['--use-gpu']
    perc_train = 0.9

    mnist = True

    train_data, train_labels, validation_data, validation_labels = \
        util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")

    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    validation_data = numpy.reshape(validation_data, (validation_data.shape[0], 28, 28, 1))


    # Creating datashufflers
    train_data_shuffler = Memory(train_data, train_labels,
                                             input_shape=[28, 28, 1],
                                             batch_size=BATCH_SIZE)

    validation_data_shuffler = Memory(validation_data, validation_labels,
                                                  input_shape=[28, 28, 1],
                                                  batch_size=VALIDATION_BATCH_SIZE)



    # Preparing the architecture
    cnn = True
    if cnn:
        architecture = Chopra(seed=SEED, fc1_output=10)
        #architecture = Lenet(seed=SEED)
        #architecture = Dummy(seed=SEED)
        loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)
        #trainer = Trainer(architecture=architecture,
        #                  loss=loss,
        #                  iterations=ITERATIONS,
        #                  analizer=ExperimentAnalizer(),
        #                  prefetch=False, temp_dir="./temp/cnn")
        #trainer.train(train_data_shuffler, validation_data_shuffler)
        #trainer.train(train_data_shuffler)
    else:
        mlp = MLP(10, hidden_layers=[15, 20])
        loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)
        trainer = Trainer(architecture=mlp, loss=loss, iterations=ITERATIONS, temp_dir="./LOGS/dnn")
        trainer.train(train_data_shuffler, validation_data_shuffler)

    # Loading
    test_data_shuffler = Memory(validation_data, validation_labels,
                                input_shape=[28, 28, 1],
                                batch_size=400)

    with tf.Session() as session:
        new_net = Chopra(seed=SEED, fc1_output=10)
        new_net.load(bob.io.base.HDF5File("./temp/cnn/model.hdf5"), shape=[400, 28, 28, 1], session=session)

        [data, labels] = test_data_shuffler.get_batch()
        print new_net(data, session)






