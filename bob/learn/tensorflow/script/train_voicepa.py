#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Pavel Korshunov <pavel.korshunov@idiap.ch>
# @date: Wed 19 Oct 23:43:22 2016


"""
Simple script that trains voicePA using Tensor flow

Usage:
  train_voicepa.py [--batch-size=<arg> --validation-batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --intermediate_model=<arg> --temp-dir=<arg> --snapshot=<arg> --use-gpu]
  train_voicepa.py -h | --help
Options:
  -h --help     Show this screen.
  --batch-size=<arg>  [default: 1]
  --validation-batch-size=<arg>   [default: 128]
  --iterations=<arg>  [default: 30000]
  --validation-interval=<arg>  [default: 100]
  --intermediate_model=<arg> [default: ""]
  --temp-dir=<arg>  [default: "cnn"]
  --snapshot=<arg>  [default: 1000]
"""

from docopt import docopt

SEED = 1980

from bob.learn.tensorflow.datashuffler import DiskAudio
from bob.learn.tensorflow.network import SimpleAudio
from bob.learn.tensorflow.loss import NegLogLoss
from bob.learn.tensorflow.trainers import Trainer, constant

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,3,2"

def main():
    args = docopt(__doc__, version='voicePa training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    VALIDATION_BATCH_SIZE = int(args['--validation-batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_INTERVAL = int(args['--validation-interval'])
    USE_GPU = args['--use-gpu']
    INTERMEDIATE_MODEL = str(args['--intermediate_model'])
    TEMP_DIR = str(args['--temp-dir'])
    TRAIN_SNAPSHOT = int(args['--snapshot'])

    import bob.db.voicepa
    db_voicepa = bob.db.voicepa.Database()
    directory = "/idiap/temp/pkorshunov/data/voicePA/"

    # Preparing train set
    train_objects = db_voicepa.objects(protocol="avspoofPA", groups="train", cls=('real', 'attack'))

    train_file_names = [o.make_path(directory=directory, extension=".wav") for o in train_objects]
    train_labels = [1 if o.is_attack() else 0 for o in train_objects]
    n_classes = len(set(train_labels))

    train_data_shuffler = DiskAudio(train_file_names, train_labels,
                                    batch_size=BATCH_SIZE,
                                    seed=SEED,
                                    out_file=os.path.join(TEMP_DIR, "train_batches.txt")
                                    )

    # Preparing validation set
    validation_objects = db_voicepa.objects(protocol="avspoofPA", groups="dev", cls=('real', 'attack'))

    validation_file_names = [o.make_path(directory=directory, extension=".wav") for o in validation_objects]
    validation_labels = [1 if o.is_attack() else 0 for o in validation_objects]

    validation_data_shuffler = DiskAudio(validation_file_names, validation_labels,
                                         batch_size=VALIDATION_BATCH_SIZE,
                                         seed=SEED,
                                         out_file=os.path.join(TEMP_DIR, "validation_batches.txt")
                                         )

    conv1_kernel_size = 300
    conv1_output = 20
    conv1_stride = 100
    fc1_output = 40
#    fc1_output = 1
    # Preparing the architecture

    dnn = SimpleAudio(conv1_kernel_size=conv1_kernel_size, conv1_output=conv1_output, conv1_stride=conv1_stride,
                      fc1_output=fc1_output,
                      default_feature_layer='fc2',
                      seed=SEED, use_gpu=USE_GPU)

    loss = NegLogLoss(tf.reduce_mean)
    learning_rate = 0.0001
    trainer = Trainer(
        architecture=dnn, loss=loss, iterations=ITERATIONS,
        optimizer=tf.train.GradientDescentOptimizer(learning_rate),
        learning_rate=constant(learning_rate, name="lr"),
        temp_dir=TEMP_DIR,
        snapshot=TRAIN_SNAPSHOT,
        validation_snapshot=VALIDATION_INTERVAL,
        prefetch=False,
        verbosity_level=1,
#        model_from_file=INTERMEDIATE_MODEL,
    )
    trainer.train(train_data_shuffler, validation_data_shuffler)
