#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


"""
Load and debug tensorflow models

Usage:
  load_and_debug.py <configuration> <output-dir>
  load_and_debug.py -h | --help
Options:
  -h --help     Show this screen.
"""

from docopt import docopt
import bob.io.base
import os
import numpy
import imp
import bob.learn.tensorflow
import tensorflow as tf

import logging
logger = logging.getLogger("bob.learn")


def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')
    output_dir = args['<output-dir>']

    config = imp.load_source('config', args['<configuration>'])

    # Cleaning all variables in case you are loading the checkpoint
    tf.reset_default_graph() if os.path.exists(output_dir) else None

    logger.info("Directory already exists, trying to get the last checkpoint")

    trainer = config.Trainer(config.train_data_shuffler,
                             iterations=0,
                             analizer=None,
                             temp_dir=output_dir)
    trainer.create_network_from_file(output_dir)
    import ipdb; ipdb.set_trace();
    
    debug=True
