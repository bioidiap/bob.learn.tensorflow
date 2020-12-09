# fmt: off
from .balanced_cross_entropy import \
    balanced_sigmoid_cross_entropy_loss_weights  # noqa: F401
from .balanced_cross_entropy import \
    balanced_softmax_cross_entropy_loss_weights  # noqa: F401
# fmt: on
from .center_loss import CenterLoss
from .center_loss import CenterLossLayer
from .pixel_wise import PixelwiseBinaryCrossentropy


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


__appropriate__(CenterLoss, CenterLossLayer, PixelwiseBinaryCrossentropy)
__all__ = [_ for _ in dir() if not _.startswith("_")]
