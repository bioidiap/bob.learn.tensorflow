# see https://docs.python.org/3/library/pkgutil.html
from .Base import Base
from .Siamese import Siamese
from .Triplet import Triplet
from .Memory import Memory
from .Disk import Disk
from .OnlineSampling import OnLineSampling

from .SiameseMemory import SiameseMemory
from .TripletMemory import TripletMemory
from .TripletWithSelectionMemory import TripletWithSelectionMemory
from .TripletWithFastSelectionDisk import TripletWithFastSelectionDisk

from .SiameseDisk import SiameseDisk
from .TripletDisk import TripletDisk
from .TripletWithSelectionDisk import TripletWithSelectionDisk

from .DataAugmentation import DataAugmentation
from .ImageAugmentation import ImageAugmentation

from .Normalizer import ScaleFactor, MeanOffset, Linear


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
    Base,
    Siamese,
    Triplet,
    Memory,
    Disk,
    OnlineSampling,
    SiameseMemory,
    TripletMemory,
    TripletWithSelectionMemory,
    TripletWithFastSelectionDisk,
    SiameseDisk,
    TripletDisk,
    TripletWithSelectionDisk,
    DataAugmentation,
    ImageAugmentation,
    ScaleFactor, MeanOffset, Linear

    )
__all__ = [_ for _ in dir() if not _.startswith('_')]