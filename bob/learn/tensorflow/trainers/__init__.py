
from .Trainer import Trainer
from .SiameseTrainer import SiameseTrainer
from .TripletTrainer import TripletTrainer
from .learning_rate import exponential_decay, constant
from .LogitsTrainer import LogitsTrainer, LogitsCenterLossTrainer
import numpy


def evaluate_softmax(data, labels, session, graph, data_node):
    """
    Evaluate the network assuming that the output layer is a softmax
    """

    predictions = numpy.argmax(session.run(
        graph,
        feed_dict={data_node: data[:]}), 1)

    return 100. * numpy.sum(predictions == labels) / predictions.shape[0]


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
    Trainer,
    SiameseTrainer,
    TripletTrainer,
    exponential_decay,
    constant,
    )
__all__ = [_ for _ in dir() if not _.startswith('_')]


