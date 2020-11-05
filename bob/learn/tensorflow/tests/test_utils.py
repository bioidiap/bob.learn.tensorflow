import numpy
import tensorflow as tf

from bob.learn.tensorflow.metrics import EmbeddingAccuracy


def test_embedding_accuracy_tensors():

    numpy.random.seed(10)
    samples_per_class = 5
    m = EmbeddingAccuracy()

    class_a = numpy.random.normal(loc=0, scale=0.1, size=(samples_per_class, 2))
    labels_a = numpy.zeros(samples_per_class)

    class_b = numpy.random.normal(loc=10, scale=0.1, size=(samples_per_class, 2))
    labels_b = numpy.ones(samples_per_class)

    data = numpy.vstack((class_a, class_b))
    labels = numpy.concatenate((labels_a, labels_b))

    data = tf.convert_to_tensor(value=data.astype("float32"))
    labels = tf.convert_to_tensor(value=labels.astype("int64"))
    m(labels, data)

    assert m.result() == 1.0
