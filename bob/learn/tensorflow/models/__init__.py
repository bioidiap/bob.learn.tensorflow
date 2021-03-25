from .alexnet import AlexNet_simplified
from .arcface import ArcFaceLayer
from .arcface import ArcFaceLayer3Penalties
from .arcface import ArcFaceModel
from .densenet import DeepPixBiS
from .densenet import DenseNet
from .densenet import densenet161  # noqa: F401
from .embedding_validation import EmbeddingValidation
from .mine import MineModel


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
    AlexNet_simplified,
    DenseNet,
    DeepPixBiS,
    MineModel,
    ArcFaceLayer,
    ArcFaceLayer3Penalties,
    ArcFaceModel,
    EmbeddingValidation,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
