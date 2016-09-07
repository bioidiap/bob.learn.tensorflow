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
from bob.learn.tensorflow.data import MemoryPairDataShuffler, TextDataShuffler
from bob.learn.tensorflow.network import Lenet
from bob.learn.tensorflow.trainers import SiameseTrainer
from bob.learn.tensorflow.loss import ContrastiveLoss
import bob.db.mobio
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
    data_shuffler = MemoryPairDataShuffler(data, labels,
                                           input_shape=[28, 28, 1],
                                           train_batch_size=BATCH_SIZE,
                                           validation_batch_size=BATCH_SIZE*1000
                                           )

    #db = bob.db.mobio.Database()
    #objects = db.objects(protocol="male")

    #labels = [o.client_id for o in objects]
    #file_names = [o.make_path(
    #    directory="/remote/lustre/2/temp/tpereira/FACEREC_EXPERIMENTS/mobio_male/lda/preprocessed",
    #    extension=".hdf5")
    #              for o in objects]
    #data_shuffler = TextDataShuffler(file_names, labels,
    #                                 input_shape=[80, 64, 1],
    #                                 train_batch_size=BATCH_SIZE,
    #                                 validation_batch_size=BATCH_SIZE*100)


    # Preparing the architecture
    lenet = Lenet(default_feature_layer="fc2")

    loss = ContrastiveLoss()
    trainer = SiameseTrainer(architecture=lenet,
                             loss=loss,
                             iterations=ITERATIONS,
                             base_lr=0.0001,
                             save_intermediate=False,
                             snapshot=VALIDATION_TEST)
    trainer.train(data_shuffler)


