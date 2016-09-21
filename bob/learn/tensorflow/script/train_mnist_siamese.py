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
from bob.learn.tensorflow.data import MemoryDataShuffler, TextDataShuffler
from bob.learn.tensorflow.network import Lenet, MLP, LenetDropout, VGG, Chopra
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
    mnist = True

    if mnist:
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

    else:
        import bob.db.atnt
        db = bob.db.atnt.Database()

        #import bob.db.mobio
        #db = bob.db.mobio.Database()

        # Preparing train set
        #train_objects = db.objects(protocol="male", groups="world")
        train_objects = db.objects(groups="world")
        train_labels = [o.client_id for o in train_objects]
        #directory = "/idiap/user/tpereira/face/baselines/eigenface/preprocessed",
        train_file_names = [o.make_path(
            directory="/idiap/group/biometric/databases/orl",
            extension=".pgm")
                            for o in train_objects]

        #train_data_shuffler = TextDataShuffler(train_file_names, train_labels,
        #                                       input_shape=[80, 64, 1],
        #                                       batch_size=BATCH_SIZE)
        train_data_shuffler = TextDataShuffler(train_file_names, train_labels,
                                               input_shape=[56, 46, 1],
                                               batch_size=BATCH_SIZE)

        # Preparing train set
        #validation_objects = db.objects(protocol="male", groups="dev")
        validation_objects = db.objects(groups="dev")
        validation_labels = [o.client_id for o in validation_objects]
        validation_file_names = [o.make_path(
            directory="/idiap/group/biometric/databases/orl",
            extension=".pgm")
                                 for o in validation_objects]

        #validation_data_shuffler = TextDataShuffler(validation_file_names, validation_labels,
        #                                           input_shape=[80, 64, 1],
        #                                            batch_size=VALIDATION_BATCH_SIZE)
        validation_data_shuffler = TextDataShuffler(validation_file_names, validation_labels,
                                                    input_shape=[56, 46, 1],
                                                    batch_size=VALIDATION_BATCH_SIZE)

    # Preparing the architecture
    n_classes = len(train_data_shuffler.possible_labels)

    cnn = True
    if cnn:

        # LENET PAPER CHOPRA
        #architecture = Chopra(default_feature_layer="fc7")
        architecture = Lenet(default_feature_layer="fc2", n_classes=n_classes, conv1_output=4, conv2_output=8,use_gpu=USE_GPU)
        #architecture = VGG(n_classes=n_classes, use_gpu=USE_GPU)

        #architecture = LenetDropout(default_feature_layer="fc2", n_classes=n_classes, conv1_output=4, conv2_output=8, use_gpu=USE_GPU)

        loss = ContrastiveLoss()
        #optimizer = tf.train.GradientDescentOptimizer(0.0001)
        trainer = SiameseTrainer(architecture=architecture,
                                 loss=loss,
                                 iterations=ITERATIONS,
                                 snapshot=VALIDATION_TEST)
        trainer.train(train_data_shuffler, validation_data_shuffler)
    else:
        mlp = MLP(n_classes, hidden_layers=[15, 20])

        loss = ContrastiveLoss()
        trainer = SiameseTrainer(architecture=mlp,
                                 loss=loss,
                                 iterations=ITERATIONS,
                                 snapshot=VALIDATION_TEST)
        trainer.train(train_data_shuffler, validation_data_shuffler)

