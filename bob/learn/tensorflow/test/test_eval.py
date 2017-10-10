from __future__ import print_function
import warnings as _warnings
import sys as _sys
import os
from tempfile import mkdtemp
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
from bob.io.base.test_utils import datafile
from bob.io.base import create_directories_safe

from bob.learn.tensorflow.utils.eval import eval_once
from bob.learn.tensorflow.utils.tfrecords import batch_data_and_labels
from bob.learn.tensorflow.script.db_to_tfrecords import main as tfrecords
from bob.bio.base.script.verify import main as verify
from bob.learn.tensorflow.script.train_generic import main as train_generic

dummy_config = datafile('dummy_verify_config.py', __name__)
DATA_SAHPE = (1, 112, 92)


# from https://stackoverflow.com/a/19299884
class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix="", prefix="tmp", dir=None):
        self._closed = False
        self.name = None  # Handle mkdtemp raising an exception
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        if self.name and not self._closed:
            try:
                self._rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                # Issue #10188: Emit a warning on stderr
                # if the directory could not be cleaned
                # up due to missing globals
                if "None" not in str(ex):
                    raise
                print("ERROR: {!r} while cleaning up {!r}".format(ex, self,),
                      file=_sys.stderr)
                return
            self._closed = True
            if _warn:
                self._warn("Implicitly cleaning up {!r}".format(self),
                           ResourceWarning)

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

    # XXX (ncoghlan): The following code attempts to make
    # this class tolerant of the module nulling out process
    # that happens during CPython interpreter shutdown
    # Alas, it doesn't actually manage it. See issue #10188
    _listdir = staticmethod(os.listdir)
    _path_join = staticmethod(os.path.join)
    _isdir = staticmethod(os.path.isdir)
    _islink = staticmethod(os.path.islink)
    _remove = staticmethod(os.remove)
    _rmdir = staticmethod(os.rmdir)
    _warn = _warnings.warn

    def _rmtree(self, path):
        # Essentially a stripped down version of shutil.rmtree.  We can't
        # use globals because they may be None'ed out at shutdown.
        for name in self._listdir(path):
            fullname = self._path_join(path, name)
            try:
                isdir = self._isdir(fullname) and not self._islink(fullname)
            except OSError:
                isdir = False
            if isdir:
                self._rmtree(fullname)
            else:
                try:
                    self._remove(fullname)
                except OSError:
                    pass
        try:
            self._rmdir(path)
        except OSError:
            pass


def _create_tfrecord(test_dir):
    config_path = os.path.join(test_dir, 'tfrecordconfig.py')
    with open(dummy_config) as f, open(config_path, 'w') as f2:
        f2.write(f.read().replace('TEST_DIR', test_dir))
    verify([config_path])
    tfrecords([config_path])
    return os.path.join(test_dir, 'sub_directory', 'dev.tfrecords')


def _create_checkpoint(checkpoint_dir, dummy_tfrecord):
    config = '''
import tensorflow as tf

checkpoint_dir = "{}"
tfrecord_filenames = ['{}']
data_shape = {}
data_type = tf.uint8
batch_size = 32
epochs = 1
learning_rate = 0.00001

from bob.learn.tensorflow.utils.tfrecords import shuffle_data_and_labels
def get_data_and_labels():
    return shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size, epochs=epochs)

def architecture(images):
    images = tf.cast(images, tf.float32)
    logits = tf.reshape(images, [-1, 92 * 112])
    logits = tf.layers.dense(inputs=logits, units=20,
                             activation=tf.nn.relu)
    return logits

def loss(logits, labels):
    predictor = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(predictor)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
'''.format(checkpoint_dir, dummy_tfrecord, DATA_SAHPE)
    create_directories_safe(checkpoint_dir)
    config_path = os.path.join(checkpoint_dir, 'config.py')
    with open(config_path, 'w') as f:
        f.write(config)
    train_generic([config_path])


def test_eval_once():
    batch_size = 2
    with tf.Graph().as_default() as graph, TemporaryDirectory() as tmpdir:

        checkpoint_dir = os.path.join(tmpdir, 'checkpoint_dir')
        dummy_tfrecord = _create_tfrecord(tmpdir)
        _create_checkpoint(checkpoint_dir, dummy_tfrecord)

        # Get images and labels
        with tf.name_scope('input'):
            images, labels = batch_data_and_labels(
                [dummy_tfrecord], DATA_SAHPE, tf.uint8, batch_size, epochs=1)

        images = tf.cast(images, tf.float32)
        logits = tf.reshape(images, [-1, 92 * 112])
        logits = tf.layers.dense(inputs=logits, units=20,
                                 activation=tf.nn.relu)

        # Calculate predictions.
        prediction_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver()
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tmpdir, graph)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        path = ckpt.model_checkpoint_path
        ret_val = eval_once(saver, summary_writer, prediction_op, summary_op,
                            path, '1', None, batch_size)
        assert ret_val == 0, str(ret_val)
