import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper

from ..utils import pdist


def predict_using_tensors(embedding, labels):
    """
    Compute the predictions through exhaustive comparisons between
    embeddings using tensors
    """

    # Fitting the main diagonal with infs (removing comparisons with the same
    # sample)
    inf = tf.cast(tf.ones_like(labels), tf.float32) * np.inf

    distances = pdist(embedding)
    distances = tf.linalg.set_diag(distances, inf)
    indexes = tf.argmin(input=distances, axis=1)
    return tf.gather(labels, indexes)


def accuracy_from_embeddings(labels, prelogits):
    labels = tf.reshape(labels, (-1,))
    embeddings = tf.nn.l2_normalize(prelogits, 1)
    predictions = predict_using_tensors(embeddings, labels)
    return tf.cast(tf.math.equal(labels, predictions), K.floatx())


class EmbeddingAccuracy(MeanMetricWrapper):
    """Calculates accuracy from labels and prelogits.
    This class relies on the fact that, in each batch, at least two images are
    available from each class(identity).
    """

    def __init__(self, name="embedding_accuracy", dtype=None, **kwargs):
        super().__init__(accuracy_from_embeddings, name, dtype=dtype, **kwargs)
