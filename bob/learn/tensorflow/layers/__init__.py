#from DataShuffler import *
from .Layer import Layer
from .Conv1D import Conv1D
from .Conv2D import Conv2D
from .FullyConnected import FullyConnected
from .MaxPooling import MaxPooling
from .AveragePooling import AveragePooling
from .Dropout import Dropout
from .InputLayer import InputLayer


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
    Layer,
    Conv1D,
    Conv2D,
    FullyConnected,
    MaxPooling,
    AveragePooling,
    Dropout,
    InputLayer,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]

