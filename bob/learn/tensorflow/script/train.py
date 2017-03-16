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
  --output-dir=<arg>    [default: ./logs/]
  --pretrained-net=<arg>
"""


from docopt import docopt
import imp


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

    trainer = config.Trainer(architecture=config.architecture,
                             loss=config.loss,
                             iterations=ITERATIONS,
                             analizer=None,
                             prefetch=PREFETCH,
                             learning_rate=config.learning_rate,
                             temp_dir=OUTPUT_DIR,
                             snapshot=100,
                             model_from_file=PRETRAINED_NET,
                             use_gpu=USE_GPU
                             )

    trainer.train(config.train_data_shuffler)
