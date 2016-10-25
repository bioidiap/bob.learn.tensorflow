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
from bob.learn.tensorflow.datashuffler import TripletWithSelectionDisk, TripletDisk, TripletWithFastSelectionDisk
from bob.learn.tensorflow.network import Lenet, MLP, LenetDropout, VGG, Chopra, Dummy, FaceNet
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
    directory = "/idiap/temp/tpereira/DEEP_FACE/CASIA_WEBFACE/mobio/preprocessed/"

    # Preparing train set
    #train_objects = db_mobio.objects(protocol="male", groups="world")
    train_objects = sorted(db_mobio.objects(protocol="male", groups="world"), key=lambda x: x.id)
    train_labels = [int(o.client_id) for o in train_objects]
    n_classes = len(set(train_labels))

    train_file_names = [o.make_path(
        directory=directory,
        extension=".hdf5")
                        for o in train_objects]
    #train_data_shuffler = TripletWithSelectionDisk(train_file_names, train_labels,
    #                                               input_shape=[56, 56, 3],
    #                                               total_identities=8,
    #                                               batch_size=BATCH_SIZE)

    train_data_shuffler = TripletWithFastSelectionDisk(train_file_names, train_labels,
                                                       input_shape=[112, 112, 3],
                                                       batch_size=BATCH_SIZE,
                                                       total_identities=8)

    # Preparing train set
    validation_objects = sorted(db_mobio.objects(protocol="male", groups="dev"), key=lambda x: x.id)
    #validation_objects = db_mobio.objects(protocol="male", groups="world")
    validation_labels = [o.client_id for o in validation_objects]

    validation_file_names = [o.make_path(
        directory=directory,
        extension=".hdf5")
                             for o in validation_objects]
    validation_data_shuffler = TripletDisk(validation_file_names, validation_labels,
                                           input_shape=[112, 112, 3],
                                           batch_size=VALIDATION_BATCH_SIZE)
    # Preparing the architecture
    architecture = Chopra(seed=SEED, fc1_output=n_classes)
    #architecture = Chopra(seed=SEED, fc1_output=n_classes)
    #architecture = FaceNet(seed=SEED, use_gpu=USE_GPU)
    #optimizer = tf.train.GradientDescentOptimizer(0.0005)


    #loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)
    #trainer = Trainer(architecture=architecture, loss=loss,
    #                  iterations=ITERATIONS,
    #                  prefetch=False,
    #                  optimizer=optimizer,
    #                  temp_dir="./LOGS/cnn")

    #loss = ContrastiveLoss(contrastive_margin=4.)
    #trainer = SiameseTrainer(architecture=architecture, loss=loss,
    #                         iterations=ITERATIONS,
    #                         prefetch=False,
    #                         optimizer=optimizer,
    #                         temp_dir="./LOGS_MOBIO/siamese-cnn-prefetch")

    loss = TripletLoss(margin=0.5)
    #optimizer = tf.train.GradientDescentOptimizer(0.000000000001)
    #optimizer = optimizer,
    trainer = TripletTrainer(architecture=architecture, loss=loss,
                             iterations=ITERATIONS,
                             base_learning_rate=0.0001,
                             prefetch=False,
                             temp_dir="./LOGS_MOBIO/triplet-cnn")

    #trainer.train(train_data_shuffler, validation_data_shuffler)
    trainer.train(train_data_shuffler)
