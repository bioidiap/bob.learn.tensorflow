from .SequenceNetwork import SequenceNetwork
from .Lenet import Lenet
from .Chopra import Chopra
from .Dummy import Dummy
from .LenetDropout import LenetDropout
from .MLP import MLP
from .FaceNet import FaceNet
from .FaceNetSimple import FaceNetSimple
from .VGG16 import VGG16
from .VGG16_mod import VGG16_mod
from .SimpleAudio import SimpleAudio
from .Embedding import Embedding
#from .Input import Input

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
    SequenceNetwork,
    Lenet,
    Chopra,
    Dummy,
    LenetDropout,
    MLP,
    FaceNet,
    FaceNetSimple,
    VGG16,
    VGG16_mod,
    SimpleAudio,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]

