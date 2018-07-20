#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST

import numpy
import tensorflow as tf
from tensorflow.python.client import device_lib


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 1))
        return d


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

    validation_data = data[n_train:n_train +
                           n_validation, :].astype("float32") * 0.00390625
    validation_labels = labels[n_train:n_train + n_validation]

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
            'data': _bytes_feature(img_raw),
            'label': _int64_feature(labels[i]),
            'key': _bytes_feature(b'-')
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def compute_eer(data_train, labels_train, data_validation, labels_validation,
                n_classes):
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
        p = [
            cosine(models[i], positive_data[j])
            for j in range(positive_data.shape[0])
        ]
        positive_scores = numpy.hstack((positive_scores, p))

        # negative scoring
        indexes = labels_validation != i
        negative_data = data_validation[indexes, :]
        n = [
            cosine(models[i], negative_data[j])
            for j in range(negative_data.shape[0])
        ]
        negative_scores = numpy.hstack((negative_scores, n))

    # Computing performance based on EER
    negative_scores = (-1) * negative_scores
    positive_scores = (-1) * positive_scores

    threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
    far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
    eer = (far + frr) / 2.

    return eer


def compute_accuracy(data_train, labels_train, data_validation,
                     labels_validation, n_classes):
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
        architecture.inference_placeholder,
        feature_layer=feature_layer,
        training=False)

    embeddings = numpy.zeros(shape=(image.shape[0], embbeding_dim))
    for i in range(image.shape[0]):
        feed_dict = {
            architecture.inference_placeholder: image[i:i + 1, :, :, :]
        }
        embedding = session.run(
            [tf.nn.l2_normalize(inference_graph, 1, 1e-10)],
            feed_dict=feed_dict)[0]
        embedding = numpy.reshape(embedding, numpy.prod(embedding.shape[1:]))
        embeddings[i] = embedding

    return embeddings


def cdist(A):
    """
    Compute a pairwise euclidean distance in the same fashion
    as in scipy.spation.distance.cdist
    """
    with tf.variable_scope('Pairwisedistance'):
        ones_1 = tf.reshape(
            tf.cast(tf.ones_like(A), tf.float32)[:, 0], [1, -1])
        p1 = tf.matmul(
            tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), ones_1)

        ones_2 = tf.reshape(
            tf.cast(tf.ones_like(A), tf.float32)[:, 0], [-1, 1])
        p2 = tf.transpose(
            tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(A), 1), shape=[-1, 1]),
                ones_2,
                transpose_b=True))

        return tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, A, transpose_b=True))


def predict_using_tensors(embedding, labels, num=None):
    """
    Compute the predictions through exhaustive comparisons between
    embeddings using tensors
    """

    # Fitting the main diagonal with infs (removing comparisons with the same
    # sample)
    inf = tf.cast(tf.ones_like(labels), tf.float32) * numpy.inf

    distances = cdist(embedding)
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
        tf.equal(p, l) for p, l in zip(
            tf.unstack(predictions, num=num), tf.unstack(labels, num=num))
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

    from scipy.spatial.distance import cdist

    distances = cdist(embedding, embedding)

    n_samples = embedding.shape[0]

    # Fitting the main diagonal with infs (removing comparisons with the same
    # sample)
    numpy.fill_diagonal(distances, numpy.inf)

    indexes = distances.argmin(axis=1)

    # Computing the argmin excluding comparisons with the same samples
    # Basically, we are excluding the main diagonal

    #valid_indexes = distances[distances>0].reshape(n_samples, n_samples-1).argmin(axis=1)

    # Getting the original positions of the indexes in the 1-axis
    #corrected_indexes = [ i if i<j else i+1 for i, j in zip(valid_indexes, range(n_samples))]

    matching = [
        labels[i] == labels[j] for i, j in zip(range(n_samples), indexes)
    ]
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
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
        raise ValueError("The image needs to be at least 3 dimensional but it "
                         "was {}".format(ndim))
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
        raise ValueError("The image needs to be at least 3 dimensional but it "
                         "was {}".format(ndim))
    axis_order = [2, 0, 1]
    shift = ndim - 3
    axis_order = list(range(ndim - 3)) + [n + shift for n in axis_order]
    return tf.transpose(image, axis_order)


to_skimage = to_matplotlib = to_channels_last
to_bob = to_channels_first
