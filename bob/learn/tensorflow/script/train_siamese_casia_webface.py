#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains CASIA WEBFACE

Usage:
  train_siamese_casia_webface.py [--batch-size=<arg> --validation-batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_siamese_casia_webface.py -h | --help
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
from bob.learn.tensorflow.datashuffler import TripletDisk, TripletWithSelectionDisk, TripletWithFastSelectionDisk
from bob.learn.tensorflow.network import Lenet, MLP, LenetDropout, VGG, Chopra, Dummy, FaceNet, FaceNetSimple, VGG16
from bob.learn.tensorflow.trainers import SiameseTrainer, TripletTrainer, constant
from bob.learn.tensorflow.loss import ContrastiveLoss, TripletLoss
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

    import bob.db.casia_webface
    db_casia = bob.db.casia_webface.Database()

    # Preparing train set
    train_objects = sorted(db_casia.objects(groups="world"), key=lambda x: x.id)
    #train_objects = db.objects(groups="world")
    train_labels = [int(o.client_id) for o in train_objects]
    directory = "/idiap/temp/tpereira/DEEP_FACE/CASIA_WEBFACE/casia_webface/preprocessed"

    train_file_names = [o.make_path(
        directory=directory,
        extension=".hdf5")
                        for o in train_objects]

    #train_data_shuffler = TripletWithFastSelectionDisk(train_file_names, train_labels,
    #                                                   input_shape=[224, 224, 3],
    #                                                   batch_size=BATCH_SIZE,
    #                                                   total_identities=16)

    train_data_shuffler = TripletDisk(train_file_names, train_labels,
                                           input_shape=[224, 224, 3],
                                           batch_size=VALIDATION_BATCH_SIZE)


    # Preparing train set
    directory = "/idiap/temp/tpereira/DEEP_FACE/CASIA_WEBFACE/mobio/preprocessed"
    validation_objects = sorted(db_mobio.objects(protocol="male", groups="dev"), key=lambda x: x.id)
    validation_labels = [o.client_id for o in validation_objects]

    validation_file_names = [o.make_path(
        directory=directory,
        extension=".hdf5")
                             for o in validation_objects]

    validation_data_shuffler = TripletDisk(validation_file_names, validation_labels,
                                           input_shape=[224, 224, 3],
                                           batch_size=VALIDATION_BATCH_SIZE)
    # Preparing the architecture
    # LENET PAPER CHOPRA
    #architecture = FaceNetSimple(seed=SEED, use_gpu=USE_GPU)
    architecture = VGG16(seed=SEED, use_gpu=USE_GPU)

    optimizer = tf.train.GradientDescentOptimizer(0.05)
    loss = TripletLoss(margin=0.2)
    trainer = TripletTrainer(architecture=architecture, loss=loss,
                             iterations=ITERATIONS,
                             learning_rate=constant(),
                             optimizer=optimizer,
                             prefetch=False,
                             temp_dir="/idiap/temp/tpereira/CNN_MODELS_GRAD_DESC/triplet-cnn-selection-gpu")

    #import ipdb; ipdb.set_trace()
    #trainer.train(train_data_shuffler, validation_data_shuffler)
    trainer.train(train_data_shuffler)
