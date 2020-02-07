#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST

import numpy
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.framework import function
import logging

logger = logging.getLogger(__name__)


@function.Defun(tf.float32, tf.float32)
def norm_grad(x, dy):
    return tf.expand_dims(dy, -1) * (
        x / (tf.expand_dims(tf.norm(x, ord=2, axis=-1), -1) + 1.0e-19)
    )


@function.Defun(tf.float32, grad_func=norm_grad)
def norm(x):
    return tf.norm(x, ord=2, axis=-1)


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    with tf.name_scope("euclidean_distance"):
        # d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
        d = norm(tf.subtract(x, y))
        return d


def pdist_safe(A, metric="sqeuclidean"):
    if metric != "sqeuclidean":
        raise NotImplementedError()
    r = tf.reduce_sum(A * A, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(A, A, transpose_b=True) + tf.transpose(r)
    return D


def cdist(A, B, metric="sqeuclidean"):
    if metric != "sqeuclidean":
        raise NotImplementedError()
    M1, M2 = tf.shape(A)[0], tf.shape(B)[0]
    # code from https://stackoverflow.com/a/43839605/1286165
    p1 = tf.matmul(
        tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), tf.ones(shape=(1, M2))
    )
    p2 = tf.transpose(
        tf.matmul(
            tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]),
            tf.ones(shape=(M1, 1)),
            transpose_b=True,
        )
    )

    D = tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)
    return D


def load_mnist(perc_train=0.9):
    numpy.random.seed(0)
    import bob.db.mnist

    db = bob.db.mnist.Database()
    raw_data = db.data()

    # data  = raw_data[0].astype(numpy.float64)
    data = raw_data[0]
    labels = raw_data[1]

    # Shuffling
    total_samples = data.shape[0]
    indexes = numpy.array(range(total_samples))
    numpy.random.shuffle(indexes)

    # Spliting train and validation
    n_train = int(perc_train * indexes.shape[0])
    n_validation = total_samples - n_train

    train_data = data[0:n_train, :].astype("float32") * 0.00390625
    train_labels = labels[0:n_train]

    validation_data = (
        data[n_train : n_train + n_validation, :].astype("float32") * 0.00390625
    )
    validation_labels = labels[n_train : n_train + n_validation]

    return train_data, train_labels, validation_data, validation_labels


def create_mnist_tfrecord(tfrecords_filename, data, labels, n_samples=6000):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for i in range(n_samples):
        img = data[i]
        img_raw = img.tostring()
        feature = {
            "data": _bytes_feature(img_raw),
            "label": _int64_feature(labels[i]),
            "key": _bytes_feature(b"-"),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def compute_eer(
    data_train, labels_train, data_validation, labels_validation, n_classes
):
    import bob.measure
    from scipy.spatial.distance import cosine

    # Creating client models
    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(numpy.mean(data_train[indexes, :], axis=0))

    # Probing
    positive_scores = numpy.zeros(shape=0)
    negative_scores = numpy.zeros(shape=0)

    for i in range(n_classes):
        # Positive scoring
        indexes = labels_validation == i
        positive_data = data_validation[indexes, :]
        p = [cosine(models[i], positive_data[j]) for j in range(positive_data.shape[0])]
        positive_scores = numpy.hstack((positive_scores, p))

        # negative scoring
        indexes = labels_validation != i
        negative_data = data_validation[indexes, :]
        n = [cosine(models[i], negative_data[j]) for j in range(negative_data.shape[0])]
        negative_scores = numpy.hstack((negative_scores, n))

    # Computing performance based on EER
    negative_scores = (-1) * negative_scores
    positive_scores = (-1) * positive_scores

    threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
    far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
    eer = (far + frr) / 2.0

    return eer


def compute_accuracy(
    data_train, labels_train, data_validation, labels_validation, n_classes
):
    from scipy.spatial.distance import cosine

    # Creating client models
    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(numpy.mean(data_train[indexes, :], axis=0))

    # Probing
    tp = 0
    for i in range(data_validation.shape[0]):

        d = data_validation[i, :]
        l = labels_validation[i]

        scores = [cosine(m, d) for m in models]
        predict = numpy.argmax(scores)

        if predict == l:
            tp += 1

    return (float(tp) / data_validation.shape[0]) * 100


def debug_embbeding(image, architecture, embbeding_dim=2, feature_layer="fc3"):
    """
    """
    import tensorflow as tf
    from bob.learn.tensorflow.utils.session import Session

    session = Session.instance(new=False).session
    inference_graph = architecture.compute_graph(
        architecture.inference_placeholder, feature_layer=feature_layer, training=False
    )

    embeddings = numpy.zeros(shape=(image.shape[0], embbeding_dim))
    for i in range(image.shape[0]):
        feed_dict = {architecture.inference_placeholder: image[i : i + 1, :, :, :]}
        embedding = session.run(
            [tf.nn.l2_normalize(inference_graph, 1, 1e-10)], feed_dict=feed_dict
        )[0]
        embedding = numpy.reshape(embedding, numpy.prod(embedding.shape[1:]))
        embeddings[i] = embedding

    return embeddings


def pdist(A):
    """
    Compute a pairwise euclidean distance in the same fashion
    as in scipy.spation.distance.pdist
    """
    with tf.name_scope("Pairwisedistance"):
        ones_1 = tf.reshape(tf.cast(tf.ones_like(A), tf.float32)[:, 0], [1, -1])
        p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), ones_1)

        ones_2 = tf.reshape(tf.cast(tf.ones_like(A), tf.float32)[:, 0], [-1, 1])
        p2 = tf.transpose(
            tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(A), 1), shape=[-1, 1]),
                ones_2,
                transpose_b=True,
            )
        )

        return tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, A, transpose_b=True))


def predict_using_tensors(embedding, labels, num=None):
    """
    Compute the predictions through exhaustive comparisons between
    embeddings using tensors
    """

    # Fitting the main diagonal with infs (removing comparisons with the same
    # sample)
    inf = tf.cast(tf.ones_like(labels), tf.float32) * numpy.inf

    distances = pdist(embedding)
    distances = tf.matrix_set_diag(distances, inf)
    indexes = tf.argmin(distances, axis=1)
    return [labels[i] for i in tf.unstack(indexes, num=num)]


def compute_embedding_accuracy_tensors(embedding, labels, num=None):
    """
    Compute the accuracy in a closed-set

    **Parameters**

    embeddings: `tf.Tensor`
      Set of embeddings

    labels: `tf.Tensor`
      Correspondent labels
    """

    # Fitting the main diagonal with infs (removing comparisons with the same
    # sample)
    predictions = predict_using_tensors(embedding, labels, num=num)
    matching = [
        tf.equal(p, l)
        for p, l in zip(tf.unstack(predictions, num=num), tf.unstack(labels, num=num))
    ]

    return tf.reduce_sum(tf.cast(matching, tf.uint8)) / len(predictions)


def compute_embedding_accuracy(embedding, labels):
    """
    Compute the accuracy in a closed-set

    **Parameters**

    embeddings: :any:`numpy.array`
      Set of embeddings

    labels: :any:`numpy.array`
      Correspondent labels
    """

    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(embedding))

    n_samples = embedding.shape[0]

    # Fitting the main diagonal with infs (removing comparisons with the same
    # sample)
    numpy.fill_diagonal(distances, numpy.inf)

    indexes = distances.argmin(axis=1)

    # Computing the argmin excluding comparisons with the same samples
    # Basically, we are excluding the main diagonal

    # valid_indexes = distances[distances>0].reshape(n_samples, n_samples-1).argmin(axis=1)

    # Getting the original positions of the indexes in the 1-axis
    # corrected_indexes = [ i if i<j else i+1 for i, j in zip(valid_indexes, range(n_samples))]

    matching = [labels[i] == labels[j] for i, j in zip(range(n_samples), indexes)]
    accuracy = sum(matching) / float(n_samples)

    return accuracy


def get_available_gpus():
    """Returns the number of GPU devices that are available.

    Returns
    -------
    [str]
        The names of available GPU devices.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def to_channels_last(image):
    """Converts the image to channel_last format. This is the same format as in
    matplotlib, skimage, and etc.

    Parameters
    ----------
    image : `tf.Tensor`
        At least a 3 dimensional image. If the dimension is more than 3, the
        last 3 dimensions are assumed to be [C, H, W].

    Returns
    -------
    image : `tf.Tensor`
        The image in [..., H, W, C] format.

    Raises
    ------
    ValueError
        If dim of image is less than 3.
    """
    ndim = len(image.shape)
    if ndim < 3:
        raise ValueError(
            "The image needs to be at least 3 dimensional but it " "was {}".format(ndim)
        )
    axis_order = [1, 2, 0]
    shift = ndim - 3
    axis_order = list(range(ndim - 3)) + [n + shift for n in axis_order]
    return tf.transpose(image, axis_order)


def to_channels_first(image):
    """Converts the image to channel_first format. This is the same format as
    in bob.io.image and bob.io.video.

    Parameters
    ----------
    image : `tf.Tensor`
        At least a 3 dimensional image. If the dimension is more than 3, the
        last 3 dimensions are assumed to be [H, W, C].

    Returns
    -------
    image : `tf.Tensor`
        The image in [..., C, H, W] format.

    Raises
    ------
    ValueError
        If dim of image is less than 3.
    """
    ndim = len(image.shape)
    if ndim < 3:
        raise ValueError(
            "The image needs to be at least 3 dimensional but it " "was {}".format(ndim)
        )
    axis_order = [2, 0, 1]
    shift = ndim - 3
    axis_order = list(range(ndim - 3)) + [n + shift for n in axis_order]
    return tf.transpose(image, axis_order)


to_skimage = to_matplotlib = to_channels_last
to_bob = to_channels_first


def bytes2human(n, format="%(value).1f %(symbol)s", symbols="customary"):
    """Convert n bytes into a human readable string based on format.
    From: https://code.activestate.com/recipes/578019-bytes-to-human-human-to-
    bytes-converter/
    Author: Giampaolo Rodola' <g.rodola [AT] gmail [DOT] com>
    License: MIT
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """
    SYMBOLS = {
        "customary": ("B", "K", "M", "G", "T", "P", "E", "Z", "Y"),
        "customary_ext": (
            "byte",
            "kilo",
            "mega",
            "giga",
            "tera",
            "peta",
            "exa",
            "zetta",
            "iotta",
        ),
        "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
        "iec_ext": (
            "byte",
            "kibi",
            "mebi",
            "gibi",
            "tebi",
            "pebi",
            "exbi",
            "zebi",
            "yobi",
        ),
    }
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def random_choice_no_replacement(one_dim_input, num_indices_to_drop=3, sort=False):
    """Similar to np.random.choice with no replacement.
    Code from https://stackoverflow.com/a/54755281/1286165
    """
    input_length = tf.shape(one_dim_input)[0]

    # create uniform distribution over the sequence
    uniform_distribution = tf.random.uniform(
        shape=[input_length],
        minval=0,
        maxval=None,
        dtype=tf.float32,
        seed=None,
        name=None,
    )

    # grab the indices of the greatest num_words_to_drop values from the distibution
    _, indices_to_keep = tf.nn.top_k(
        uniform_distribution, input_length - num_indices_to_drop
    )

    # sort the indices
    if sort:
        sorted_indices_to_keep = tf.sort(indices_to_keep)
    else:
        sorted_indices_to_keep = indices_to_keep

    # gather indices from the input array using the filtered actual array
    result = tf.gather(one_dim_input, sorted_indices_to_keep)
    return result
