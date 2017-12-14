from .BaseLoss import mean_cross_entropy_loss, mean_cross_entropy_center_loss
from .ContrastiveLoss import contrastive_loss, contrastive_loss_deprecated
from .TripletLoss import triplet_loss, triplet_average_loss, triplet_fisher_loss, triplet_loss_deprecated

#from .NegLogLoss import NegLogLoss


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


__appropriate__(mean_cross_entropy_loss, mean_cross_entropy_center_loss,
                contrastive_loss, triplet_loss, triplet_average_loss,
                triplet_fisher_loss)
__all__ = [_ for _ in dir() if not _.startswith('_')]
