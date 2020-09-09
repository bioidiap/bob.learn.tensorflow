# from .BaseLoss import mean_cross_entropy_loss, mean_cross_entropy_center_loss
from .center_loss import CenterLoss
from .ContrastiveLoss import contrastive_loss
from .mmd import *
from .pairwise_confusion import total_pairwise_confusion
from .pixel_wise import PixelWise
from .StyleLoss import content_loss
from .StyleLoss import denoising_loss
from .StyleLoss import linear_gram_style_loss
from .TripletLoss import triplet_average_loss
from .TripletLoss import triplet_fisher_loss
from .TripletLoss import triplet_loss
from .utils import *
from .vat import VATLoss


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
    # mean_cross_entropy_loss,
    # mean_cross_entropy_center_loss,
    contrastive_loss,
    triplet_loss,
    triplet_average_loss,
    triplet_fisher_loss,
    VATLoss,
    PixelWise,
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
