#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains CASIA WEBFACE

Usage:
  train_mobio.py [--batch-size=<arg> --validation-batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_mobio.py -h | --help
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
from bob.learn.tensorflow.trainers import SiameseTrainer, Trainer, TripletTrainer
from bob.learn.tensorflow.loss import ContrastiveLoss, BaseLoss, TripletLoss
import numpy


def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    VALIDATION_BATCH_SIZE = int(args['--validation-batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    USE_GPU = args['--use-gpu']
    perc_train = 0.9

    import bob.db.mobio
    db_mobio = bob.db.mobio.Database()
    directory = "/idiap/temp/tpereira/DEEP_FACE/CASIA/preprocessed"

    # Preparing train set
    #train_objects = db_mobio.objects(protocol="male", groups="world")
    train_objects = db_mobio.objects(protocol="male", groups="dev")
    train_labels = [int(o.client_id) for o in train_objects]
    n_classes = len(set(train_labels))

    train_file_names = [o.make_path(
        directory=directory,
        extension=".hdf5")
                        for o in train_objects]
    train_data_shuffler = TextDataShuffler(train_file_names, train_labels,
                                           input_shape=[125, 125, 3],
                                           batch_size=BATCH_SIZE)

    # Preparing train set
    validation_objects = db_mobio.objects(protocol="male", groups="dev")
    #validation_objects = db_mobio.objects(protocol="male", groups="world")
    validation_labels = [o.client_id for o in validation_objects]

    validation_file_names = [o.make_path(
        directory=directory,
        extension=".hdf5")
                             for o in validation_objects]
    validation_data_shuffler = TextDataShuffler(validation_file_names, validation_labels,
                                                input_shape=[125, 125, 3],
                                                batch_size=VALIDATION_BATCH_SIZE)
    # Preparing the architecture
    #architecture = Chopra(seed=SEED, fc1_output=n_classes)
    architecture = Chopra(seed=SEED)
    optimizer = tf.train.GradientDescentOptimizer(0.00000001)


    #loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)
    #trainer = Trainer(architecture=architecture, loss=loss,
    #                  iterations=ITERATIONS,
    #                  prefetch=False,
    #                  optimizer=optimizer,
    #                  temp_dir="./LOGS/cnn")

    #loss = ContrastiveLoss(contrastive_margin=4.)
    #trainer = SiameseTrainer(architecture=architecture, loss=loss,
    #                         iterations=ITERATIONS,
    #                         prefetch=True,
    #                         optimizer=optimizer,
    #                         temp_dir="./LOGS_MOBIO/siamese-cnn-prefetch")

    loss = TripletLoss(margin=4.)
    trainer = TripletTrainer(architecture=architecture, loss=loss,
                             iterations=ITERATIONS,
                             prefetch=True,
                             optimizer=optimizer,
                             temp_dir="./LOGS_MOBIO/triplet-cnn-prefetch")

    trainer.train(train_data_shuffler, validation_data_shuffler)
