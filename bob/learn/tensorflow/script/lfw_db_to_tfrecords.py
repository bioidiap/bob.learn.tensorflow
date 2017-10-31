#!/usr/bin/env python

"""Script that converts bob.db.lfw database to TF records

Usage:
  %(prog)s <data-path> <output-file> [--extension=<arg> --protocol=<arg> --data-type=<arg> --verbose]
  %(prog)s --help
  %(prog)s --version

Options:
  -h --help  show this help message and exit
  <data-path>          Path that contains the features
  --extension=<arg>    Default feature extension   [default: .hdf5]
  --protocol=<arg>     One of the LFW protocols    [default: view1]
  --data-type=<arg>    TFRecord data type [default: uint8]


The possible protocol options are the following:
  'view1', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10'

More details about our interface to LFW database can be found in
https://www.idiap.ch/software/bob/docs/bob/bob.db.lfw/master/index.html.


"""

import tensorflow as tf
from bob.io.base import create_directories_safe
from bob.bio.base.utils import load, read_config_file
from bob.core.log import setup, set_verbosity_level
import bob.db.lfw
import os
import bob.io.image
import bob.io.base
import numpy

logger = setup(__name__)


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def file_to_label(client_ids, f):
    return client_ids[str(f.client_id)]

def get_pairs(all_pairs, match=True):

    enroll = []
    probe = []
    for p in all_pairs:
        if p.is_match == match:
            enroll.append(p.enroll_file)
            probe.append(p.probe_file)

    return enroll, probe


def main(argv=None):
    from docopt import docopt
    args = docopt(__doc__, version='')

    data_path   = args['<data-path>']
    output_file = args['<output-file>']
    extension   = args['--extension']
    protocol    = args['--protocol']
    data_type   = args['--data-type']
    
    # Sets-up logging
    if args['--verbose']:
        verbosity = 2
        set_verbosity_level(logger, verbosity)

    # Loading LFW models
    database = bob.db.lfw.Database()
    enroll, probe = get_pairs(database.pairs(protocol=protocol), match=True)
    #client_ids = list(set([f.client_id for f in all_pairs]))
    
    client_ids = list(set([f.client_id for f in enroll] + [f.client_id for f in probe]))   
    client_ids = dict(zip(client_ids, range(len(client_ids))))

    create_directories_safe(os.path.dirname(output_file))

    n_files = len(enroll)
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for e, p, i in zip(enroll, probe, range(len(enroll)) ):
        logger.info('Processing pair %d out of %d', i + 1, n_files)
        
        if os.path.exists(e.make_path(data_path, extension)) and os.path.exists(p.make_path(data_path, extension)):
            for f in [e, p]:
                path = f.make_path(data_path, extension)
                data = bob.io.image.to_matplotlib(bob.io.base.load(path)).astype(data_type)
                data = data.tostring()

                feature = {'data': _bytes_feature(data),
                           'label': _int64_feature(file_to_label(client_ids, f)),
                           'key': _bytes_feature(str(f.path)),
                           }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        else:
            logger.debug("... Processing original data file '{0}' was not successful".format(path))

if __name__ == '__main__':
  main()
