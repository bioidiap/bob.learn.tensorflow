from .Chopra import Chopra
from .LightCNN9 import LightCNN9
from .Dummy import Dummy
from .MLP import MLP
from .Embedding import Embedding
from .lstm import simple_lstm_network, double_lstm_network, triple_lstm_network
from .lstm import RegularizedLoss
from .simplemlp import mlp_network
from .simplecnn import simple2Dcnn_network

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
    Chopra,
    LightCNN9,
    Dummy,
    MLP,
    RegularizedLoss,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]

