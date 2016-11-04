# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .SequenceNetwork import SequenceNetwork
from .Lenet import Lenet
from .Chopra import Chopra
from .Dummy import Dummy
from .VGG import VGG
from .LenetDropout import LenetDropout
from .MLP import MLP
from .FaceNet import FaceNet
from .FaceNetSimple import FaceNetSimple
from .VGG16 import VGG16
from .VGG16_mod import VGG16_mod

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
