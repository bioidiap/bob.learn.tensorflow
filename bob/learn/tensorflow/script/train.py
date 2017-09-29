#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 04 Jan 2017 18:00:36 CET

"""
Train a Neural network using bob.learn.tensorflow

Usage:
  train.py [--iterations=<arg> --validation-interval=<arg> --output-dir=<arg> ] <configuration> [grid --n-jobs=<arg> --job-name=<job-name> --queue=<arg>]
  train.py -h | --help

Options:
  -h --help                     Show this screen.
  --iterations=<arg>            Number of iteratiosn [default: 1000]
  --validation-interval=<arg>   Validata every n iteratiosn [default: 500]
  --output-dir=<arg>            If the directory exists, will try to get the last checkpoint [default: ./logs/]
  --n-jobs=<arg>                Number of jobs submitted to the grid [default: 3]
  --job-name=<arg>              Job name  [default: TF]
  --queue=<arg>                 SGE queue name  [default: q_gpu]
"""

from docopt import docopt
import imp
import bob.learn.tensorflow
import tensorflow as tf
import os
import sys

import logging
logger = logging.getLogger("bob.learn")


def dump_commandline():

    command_line = []
    for command in sys.argv:
        if command == "grid":
            break
        command_line.append(command)
    return command_line


def main():
    args = docopt(__doc__, version='Train Neural Net')
    
    output_dir = str(args['--output-dir'])
    iterations = int(args['--iterations'])

    grid = int(args['grid'])
    if grid:
        # Submitting jobs to SGE
        jobs = int(args['--n-jobs'])
        job_name = args['--job-name']
        queue = args['--queue']
        import gridtk

        job_manager = gridtk.sge.JobManagerSGE()
        command = dump_commandline()
        dependencies = []
        total_jobs = []
        
        kwargs = {"env": ["LD_LIBRARY_PATH=/idiap/user/tpereira/cuda/cuda-8.0/lib64:/idiap/user/tpereira/cuda/cudnn-8.0-linux-x64-v5.1/lib64:/idiap/user/tpereira/cuda/cuda-8.0/bin"]}
        
        for i in range(jobs):
            job_id = job_manager.submit(command, queue=queue, dependencies=dependencies,
                                        name=job_name + "{0}".format(i), **kwargs)
                                        
            dependencies = [job_id]
            total_jobs.append(job_id)

        logger.info("Submitted the jobs {0}".format(total_jobs))
        return True

    config = imp.load_source('config', args['<configuration>'])

    # Cleaning all variables in case you are loading the checkpoint
    tf.reset_default_graph() if os.path.exists(output_dir) else None

    # One graph trainer
    trainer = config.Trainer(config.train_data_shuffler,
                             iterations=iterations,
                             analizer=None,
                             temp_dir=output_dir)
    if os.path.exists(output_dir):
        logger.info("Directory already exists, trying to get the last checkpoint")
        trainer.create_network_from_file(output_dir)
    else:

        # Either bootstrap from scratch or take the pointer directly from the config script
        train_graph = None
        validation_graph = None
        
        if hasattr(config, 'train_graph'):
            train_graph = config.train_graph
            if hasattr(config, 'validation_graph'):
                validation_graph = config.validation_graph
            
        else:
            # Preparing the architecture
            input_pl = config.train_data_shuffler("data", from_queue=False)
            if isinstance(trainer, bob.learn.tensorflow.trainers.SiameseTrainer):
                train_graph = dict()
                train_graph['left'] = config.architecture(input_pl['left'])
                train_graph['right'] = config.architecture(input_pl['right'], reuse=True)

            elif isinstance(trainer, bob.learn.tensorflow.trainers.TripletTrainer):
                train_graph = dict()
                train_graph['anchor'] = config.architecture(input_pl['anchor'])
                train_graph['positive'] = config.architecture(input_pl['positive'], reuse=True)
                train_graph['negative'] = config.architecture(input_pl['negative'], reuse=True)
            else:
                train_graph = config.architecture(input_pl)

        trainer.create_network_from_scratch(train_graph,
                                            validation_graph=validation_graph,
                                            loss=config.loss,
                                            learning_rate=config.learning_rate,
                                            optimizer=config.optimizer)
    trainer.train()

