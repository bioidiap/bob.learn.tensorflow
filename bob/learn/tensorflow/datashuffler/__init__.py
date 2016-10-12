# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .Base import Base
from .Siamese import Siamese
from .Memory import Memory
from .SiameseMemory import SiameseMemory
from .TripletMemory import TripletMemory

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
