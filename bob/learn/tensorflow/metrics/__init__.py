from .embedding_accuracy import EmbeddingAccuracy
from .embedding_accuracy import predict_using_tensors  # noqa: F401
from .pixel_wise import PixelwiseBinaryAccuracy
from .pixel_wise import pixel_wise_binary_accuracy  # noqa: F401


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


__appropriate__(EmbeddingAccuracy, PixelwiseBinaryAccuracy)
__all__ = [_ for _ in dir() if not _.startswith("_")]
