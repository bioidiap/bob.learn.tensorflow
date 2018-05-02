from .Chopra import chopra
from .LightCNN9 import light_cnn9
from .Dummy import dummy
from .MLP import mlp
from .InceptionResnetV2 import inception_resnet_v2, inception_resnet_v2_batch_norm
from .InceptionResnetV1 import inception_resnet_v1, inception_resnet_v1_batch_norm
from . import SimpleCNN


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
    chopra,
    light_cnn9,
    dummy,
    mlp,
)

__all__ = [_ for _ in dir() if not _.startswith('_')]
