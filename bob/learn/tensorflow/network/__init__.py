from .Chopra import chopra
from .LightCNN9 import light_cnn9
from .LightCNN29 import LightCNN29
from .Dummy import Dummy
from .MLP import MLP
from .Embedding import Embedding
from .InceptionResnetV2 import inception_resnet_v2
from .InceptionResnetV1 import inception_resnet_v1


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
    light_cnn9,
    Dummy,
    MLP,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]

