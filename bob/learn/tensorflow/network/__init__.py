# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .SequenceNetwork import SequenceNetwork
from .Lenet import Lenet

