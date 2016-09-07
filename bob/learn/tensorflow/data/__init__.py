# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .BaseDataShuffler import BaseDataShuffler
from .MemoryDataShuffler import MemoryDataShuffler
from .MemoryPairDataShuffler import MemoryPairDataShuffler
from .TextDataShuffler import TextDataShuffler

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
