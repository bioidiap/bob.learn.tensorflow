#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Amir Mohammadi <amir.mohammadi@idiap.ch>

import tensorflow as tf


def balanced_softmax_cross_entropy_loss_weights(labels, dtype="float32"):
    """Computes weights that normalizes your loss per class.

    Labels must be a batch of one-hot encoded labels. The function takes labels and
    computes the weights per batch. Weights will be smaller for classes that have more
    samples in this batch. This is useful if you unbalanced classes in your dataset or
    batch.

    Parameters
    ----------
    labels : ``tf.Tensor``
        Labels of your current input. The shape must be [batch_size, n_classes]. If your
        labels are not one-hot encoded, you can use ``tf.one_hot`` to convert them first
        before giving them to this function.
    dtype : ``tf.dtype``
        The dtype that weights will have. It should be float. Best is to provide
        logits.dtype as input.

    Returns
    -------
    ``tf.Tensor``
        Computed weights that will cancel your dataset imbalance per batch.

    Examples
    --------
    >>> import numpy
    >>> import tensorflow as tf
    >>> from bob.learn.tensorflow.loss import balanced_softmax_cross_entropy_loss_weights
    >>> labels = numpy.array([[1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [1, 0, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0]], dtype="int32")
    >>> session = tf.Session() # Eager execution is also possible check https://www.tensorflow.org/guide/eager
    >>> session.run(tf.reduce_sum(labels, axis=0))
    array([20,  5,  7], dtype=int32)
    >>> session.run(balanced_softmax_cross_entropy_loss_weights(labels, dtype='float32'))
    array([0.53333336, 0.53333336, 1.5238096 , 2.1333334 , 1.5238096 ,
           0.53333336, 0.53333336, 1.5238096 , 0.53333336, 0.53333336,
           0.53333336, 0.53333336, 0.53333336, 0.53333336, 2.1333334 ,
           0.53333336, 2.1333334 , 0.53333336, 1.5238096 , 1.5238096 ,
           0.53333336, 1.5238096 , 0.53333336, 0.53333336, 2.1333334 ,
           0.53333336, 0.53333336, 0.53333336, 2.1333334 , 0.53333336,
           1.5238096 , 0.53333336], dtype=float32)

    You would use it like this:

    >>> #weights = balanced_softmax_cross_entropy_loss_weights(labels, dtype=logits.dtype)
    >>> #loss = tf.losses.softmax_cross_entropy(logits=logits, labels=labels, weights=weights)
    """
    shape = tf.cast(tf.shape(labels), dtype=dtype)
    batch_size, n_classes = shape[0], shape[1]
    weights = tf.cast(tf.reduce_sum(labels, axis=0), dtype=dtype)
    weights = batch_size / weights / n_classes
    weights = tf.gather(weights, tf.argmax(labels, axis=1))
    return weights


def balanced_sigmoid_cross_entropy_loss_weights(labels, dtype="float32"):
    """Computes weights that normalizes your loss per class.

    Labels must be a batch of binary labels. The function takes labels and
    computes the weights per batch. Weights will be smaller for the class that have more
    samples in this batch. This is useful if you unbalanced classes in your dataset or
    batch.

    Parameters
    ----------
    labels : ``tf.Tensor``
        Labels of your current input. The shape must be [batch_size] and values must be
        either 0 or 1.
    dtype : ``tf.dtype``
        The dtype that weights will have. It should be float. Best is to provide
        logits.dtype as input.

    Returns
    -------
    ``tf.Tensor``
        Computed weights that will cancel your dataset imbalance per batch.

    Examples
    --------
    >>> import numpy
    >>> import tensorflow as tf
    >>> from bob.learn.tensorflow.loss import balanced_sigmoid_cross_entropy_loss_weights
    >>> labels = numpy.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    ...                 1, 1, 0, 1, 1, 1, 0, 1, 0, 1], dtype="int32")
    >>> sum(labels), len(labels)
    (20, 32)
    >>> session = tf.Session() # Eager execution is also possible check https://www.tensorflow.org/guide/eager
    >>> session.run(balanced_sigmoid_cross_entropy_loss_weights(labels, dtype='float32'))
    array([0.8      , 0.8      , 1.3333334, 1.3333334, 1.3333334, 0.8      ,
           0.8      , 1.3333334, 0.8      , 0.8      , 0.8      , 0.8      ,
           0.8      , 0.8      , 1.3333334, 0.8      , 1.3333334, 0.8      ,
           1.3333334, 1.3333334, 0.8      , 1.3333334, 0.8      , 0.8      ,
           1.3333334, 0.8      , 0.8      , 0.8      , 1.3333334, 0.8      ,
           1.3333334, 0.8      ], dtype=float32)

    You would use it like this:

    >>> #weights = balanced_sigmoid_cross_entropy_loss_weights(labels, dtype=logits.dtype)
    >>> #loss = tf.losses.sigmoid_cross_entropy(logits=logits, labels=labels, weights=weights)
    """
    labels = tf.cast(labels, dtype='int32')
    batch_size = tf.cast(tf.shape(labels)[0], dtype=dtype)
    weights = tf.cast(tf.reduce_sum(labels), dtype=dtype)
    weights = tf.convert_to_tensor([batch_size - weights, weights])
    weights = batch_size / weights / 2
    weights = tf.gather(weights, labels)
    return weights
