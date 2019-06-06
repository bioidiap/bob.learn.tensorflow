#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import tensorflow as tf

from ..utils import get_trainable_variables, check_features
from .utils import MovingAverageOptimizer, learning_rate_decay_fn
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


__appropriate__(
    Logits, LogitsCenterLoss, Siamese, Triplet, Regressor, MovingAverageOptimizer
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
