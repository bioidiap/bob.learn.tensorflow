from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
    from tensorflow.python.estimator.estimator import \
        _load_global_step_from_checkpoint_dir
    global_step = _load_global_step_from_checkpoint_dir(path)
    return global_step
