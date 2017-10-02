#!/usr/bin/env python

"""Script that converts bob.db.lfw database to TF records

Usage:
  %(prog)s <data-path> <output-file> [--extension=<arg> --protocol=<arg> --verbose]
  %(prog)s --help
  %(prog)s --version

Options:
  -h --help  show this help message and exit
  <data-path>          Path that contains the features
  --extension=<arg>    Default feature extension   [default: .hdf5]
  --protocol=<arg>     One of the LFW protocols    [default: view1]


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

    pairs = []
    for p in all_pairs:
        if p.is_match == match:
            pairs.append(p.enroll_file)
            pairs.append(p.probe_file)

    return pairs

def bob2skimage(bob_image):
    """
    Convert bob color image to the skcit image
    """

    skimage = numpy.zeros(shape=(bob_image.shape[1], bob_image.shape[2], bob_image.shape[0]))
    skimage[:, :, 2] = bob_image[0, :, :]
    skimage[:, :, 1] = bob_image[1, :, :]    
    skimage[:, :, 0] = bob_image[2, :, :]

    return skimage


def main(argv=None):
    from docopt import docopt
    args = docopt(__doc__, version='')

    data_path   = args['<data-path>']
    output_file = args['<output-file>']
    extension   = args['--extension']
    protocol    = args['--protocol']
    
    #Setting the reader
    reader = bob.io.base.load

    # Sets-up logging
    if args['--verbose']:
        verbosity = 2
        set_verbosity_level(logger, verbosity)

    # Loading LFW models
    database = bob.db.lfw.Database()
    all_pairs = get_pairs(database.pairs(protocol=protocol), match=True)
    client_ids = list(set([f.client_id for f in all_pairs]))
    client_ids = dict(zip(client_ids, range(len(client_ids))))
    
    create_directories_safe(os.path.dirname(output_file))

    n_files = len(all_pairs)
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for i, f in enumerate(all_pairs):
        logger.info('Processing file %d out of %d', i + 1, n_files)

        path = f.make_path(data_path, extension)
        #data = reader(path).astype('uint8').tostring()
        img = bob2skimage(reader(path)).astype('float32')
        data = img.tostring()


        feature = {'train/data': _bytes_feature(data),
                   'train/label': _int64_feature(file_to_label(client_ids, f))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


if __name__ == '__main__':
  main()
