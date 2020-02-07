"""Helps training reproducible networks.
"""
import os
import random as rn
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


def set_seed(
    seed=0,
    python_hash_seed=0,
    log_device_placement=False,
    allow_soft_placement=False,
    arithmetic_optimization=None,
    allow_growth=None,
    memory_optimization=None,
):
    """Sets the seeds in python, numpy, and tensorflow in order to help
    training reproducible networks.

    Parameters
    ----------
    seed : :obj:`int`, optional
        The seed to set.
    python_hash_seed : :obj:`int`, optional
        https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    log_device_placement : :obj:`bool`, optional
        Optionally, log device placement of tensorflow variables.

    Returns
    -------
    :any:`tf.ConfigProto`
        Session config.
    :any:`tf.estimator.RunConfig`
        A run config to help training estimators.

    Notes
    -----
        This functions return a list and its length might change. Please use
        indices to select one of returned values. For example
        ``sess_config, run_config = set_seed()[:2]``.
    """
    # reproducible networks
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
    os.environ["PYTHONHASHSEED"] = "{}".format(python_hash_seed)

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(seed)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see:
    # https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement,
    )

    off = rewriter_config_pb2.RewriterConfig.OFF
    if arithmetic_optimization == "off":
        session_config.graph_options.rewrite_options.arithmetic_optimization = off

    if memory_optimization == "off":
        session_config.graph_options.rewrite_options.memory_optimization = off

    if allow_growth is not None:
        session_config.gpu_options.allow_growth = allow_growth
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(seed)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_config)
    # keras.backend.set_session(sess)

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(session_config=session_config)
    run_config = run_config.replace(tf_random_seed=seed)

    return [session_config, run_config, None, None, None]
