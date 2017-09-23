#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 04 Jan 2017 18:00:36 CET

"""
Train a Neural network using bob.learn.tensorflow

Usage:
  train.py [--iterations=<arg> --validation-interval=<arg> --output-dir=<arg> ] <configuration> [grid <jobs>]
  train.py -h | --help

Options:
  -h --help                     Show this screen.
  --iterations=<arg>            Number of iteratiosn [default: 1000]
  --validation-interval=<arg>   Validata every n iteratiosn [default: 500]
  --output-dir=<arg>            If the directory exists, will try to get the last checkpoint [default: ./logs/]
"""

from docopt import docopt
import imp
import bob.learn.tensorflow
import tensorflow as tf
import os
import sys


def dump_commandline():

    command_line = ""
    for command in sys.argv:
        if command == "grid":
            break
        command_line += "{0} ".format(command)
    return command_line


def main():
    args = docopt(__doc__, version='Train Neural Net')

    output_dir = str(args['--output-dir'])
    iterations = int(args['--iterations'])

    grid = int(args['grid'])
    if grid:
        jobs = int(args['<jobs>'])
        import gridtk

        job_manager = gridtk.sge.JobManagerSGE()
        command = dump_commandline()
        dependencies = []

        for i in range(jobs):
            job_id = job_manager.submit(command, dependencies=dependencies)
            dependencies = [job_id]

    config = imp.load_source('config', args['<configuration>'])

    # Cleaning all variables in case you are loading the checkpoint
    tf.reset_default_graph() if os.path.exists(output_dir) else None

    # One graph trainer
    trainer = config.Trainer(config.train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=output_dir)
    if os.path.exists(output_dir):
        print("Directory already exists, trying to get the last checkpoint")
        trainer.create_network_from_file(output_dir)
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

