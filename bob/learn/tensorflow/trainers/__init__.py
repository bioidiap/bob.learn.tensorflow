# see https://docs.python.org/3/library/pkgutil.html
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .Trainer import Trainer
from .SiameseTrainer import SiameseTrainer
from .TripletTrainer import TripletTrainer

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
__all__ = [_ for _ in dir() if not _.startswith('_')]


