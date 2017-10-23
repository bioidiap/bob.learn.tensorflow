#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

def check_features(features):
    if not 'data' in features.keys() or not 'key' in features.keys():
        raise ValueError("The input function needs to contain a dictionary with the keys `data` and `key` ")
    return True


from .Logits import Logits, LogitsCenterLoss
from .Siamese import Siamese
from .Triplet import Triplet


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
  """Says object was actually declared here, an not on the import module.

  Parameters:

    *args: An iterable of objects to modify

  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """

  for obj in args: obj.__module__ = __name__

__appropriate__(
    Logits,
    LogitsCenterLoss,
    Siamese,
    Triplet
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]


