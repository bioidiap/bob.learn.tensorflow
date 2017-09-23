#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 04 Jan 2017 18:00:36 CET

"""
Train a Neural network using bob.learn.tensorflow

Usage:
  train.py [--iterations=<arg> --validation-interval=<arg> --output-dir=<arg> --pretrained-net=<arg> --use-gpu --prefetch ] <configuration>
  train.py -h | --help
Options:
  -h --help     Show this screen.
  --iterations=<arg>   [default: 1000]
  --validation-interval=<arg>   [default: 100]
  --output-dir=<arg>    If the directory exists, will try to get the last checkpoint [default: ./logs/]
  --pretrained-net=<arg>
"""


from docopt import docopt
import imp
import bob.learn.tensorflow
import tensorflow as tf
import os

def main():
    args = docopt(__doc__, version='Train Neural Net')

    USE_GPU = args['--use-gpu']
    OUTPUT_DIR = str(args['--output-dir'])
    PREFETCH = args['--prefetch']
    ITERATIONS = int(args['--iterations'])

    PRETRAINED_NET = ""
    if not args['--pretrained-net'] is None:
        PRETRAINED_NET = str(args['--pretrained-net'])


    config = imp.load_source('config', args['<configuration>'])

    # One graph trainer
    trainer = config.Trainer(config.train_data_shuffler,
                             iterations=ITERATIONS,
                             analizer=None,
                             temp_dir=OUTPUT_DIR)


    if os.path.exists(OUTPUT_DIR):
        print("Directory already exists, trying to get the last checkpoint")
        import ipdb; ipdb.set_trace();
        trainer.create_network_from_file(OUTPUT_DIR)

    else:

        # Preparing the architecture
        input_pl = config.train_data_shuffler("data", from_queue=False)
        if isinstance(trainer, bob.learn.tensorflow.trainers.SiameseTrainer):
            graph = dict()
            graph['left'] = config.architecture(input_pl['left'])
            graph['right'] = config.architecture(input_pl['right'], reuse=True)

        elif isinstance(trainer, bob.learn.tensorflow.trainers.TripletTrainer):
            graph = dict()
            graph['anchor'] = config.architecture(input_pl['anchor'])
            graph['positive'] = config.architecture(input_pl['positive'], reuse=True)
            graph['negative'] = config.architecture(input_pl['negative'], reuse=True)
        else:
            graph = config.architecture(input_pl)

        trainer.create_network_from_scratch(graph, loss=config.loss,
                                            learning_rate=config.learning_rate,
                                            optimizer=config.optimizer)
    trainer.train()

