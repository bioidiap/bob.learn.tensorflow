from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def get_global_step(path):
    """Returns the global number associated with the model checkpoint path. The
    checkpoint must have been saved with the
    :any:`tf.train.MonitoredTrainingSession`.

    Parameters
    ----------
    path : str
        The path to model checkpoint, usually ckpt.model_checkpoint_path

    Returns
    -------
    global_step : int
        The global step number.
    """
    checkpoint_reader = tf.train.NewCheckpointReader(path)
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
