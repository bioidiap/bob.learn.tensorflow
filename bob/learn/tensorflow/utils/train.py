import tensorflow as tf


def check_features(features):
    if "data" not in features or "key" not in features:
        raise ValueError(
            "The input function needs to contain a dictionary with the keys `data` and `key` "
        )
    return True


def get_trainable_variables(extra_checkpoint, mode=tf.estimator.ModeKeys.TRAIN):
    """
    Given the extra_checkpoint dictionary provided to the estimator,
    extract the content of "trainable_variables".

    If trainable_variables is not provided, all end points are trainable by
    default.
    If trainable_variables==[], all end points are NOT trainable.
    If trainable_variables contains some end_points, ONLY these endpoints will
    be trainable.

    Attributes
    ----------

    extra_checkpoint: dict
      The extra_checkpoint dictionary provided to the estimator

    mode:
        The estimator mode. TRAIN, EVAL, and PREDICT. If not TRAIN, None is
        returned.

    Returns
    -------

    Returns `None` if **trainable_variables** is not in extra_checkpoint;
    otherwise returns the content of extra_checkpoint .
    """
    if mode != tf.estimator.ModeKeys.TRAIN:
        return None

    # If you don't set anything, everything is trainable
    if extra_checkpoint is None or "trainable_variables" not in extra_checkpoint:
        return None

    return extra_checkpoint["trainable_variables"]
