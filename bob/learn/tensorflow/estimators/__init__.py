#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf


def check_features(features):
    if not 'data' in features.keys() or not 'key' in features.keys():
        raise ValueError(
            "The input function needs to contain a dictionary with the keys `data` and `key` "
        )
    return True


def get_trainable_variables(extra_checkpoint,
                            mode=tf.estimator.ModeKeys.TRAIN):
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


from .Logits import Logits, LogitsCenterLoss
from .Siamese import Siamese
from .Triplet import Triplet
from .Regressor import Regressor


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, an not on the import module.

  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(Logits, LogitsCenterLoss, Siamese, Triplet, Regressor)
__all__ = [_ for _ in dir() if not _.startswith('_')]
