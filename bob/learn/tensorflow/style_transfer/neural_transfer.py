#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


import tensorflow as tf
import numpy
import os

def compute_features(input_image, architecture, checkpoint_dir, target_end_points, preprocess_fn=None):
    """
    For a given set of end_points, convolve the input image until these points

    Parameters
    ----------

    input_image: numpy.array
        Input image in the format WxHxC

    architecture:
        Pointer to the architecture function

    checkpoint_dir: str
        DCNN checkpoint directory

    end_points: dict
       Dictionary containing the end point tensors

    preprocess_fn:
       Pointer to a preprocess function

    """

    input_pl = tf.placeholder('float32', shape=(1, input_image.shape[1],
                                                   input_image.shape[2],
                                                   input_image.shape[3]))

    if preprocess_fn is None:
        _, end_points = architecture(input_pl, mode=tf.estimator.ModeKeys.PREDICT, trainable_variables=None)
    else:
        _, end_points = architecture(tf.stack([preprocess_fn(i) for i in tf.unstack(input_pl)]), mode=tf.estimator.ModeKeys.PREDICT, trainable_variables=None)
    with tf.Session() as sess:
        # Restoring the checkpoint for the given architecture
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if os.path.isdir(checkpoint_dir):
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        else:
            saver.restore(sess, checkpoint_dir)

        #content_feature = sess.run(end_points[CONTENT_END_POINTS], feed_dict={input_image: content_image})
        features = []
        for ep in target_end_points:
            feature = sess.run(end_points[ep], feed_dict={input_pl: input_image})
            features.append(feature)

    # Killing the graph
    tf.reset_default_graph()
    return features


def compute_gram(features):
    """
    Given a list of features (as numpy.arrays) comput the gram matrices of each
    pinning the channel as in:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    Parameters
    ----------

    features: numpy.array
      Convolved features in the format NxWxHxC

    """

    grams = []
    for f in features:
        f = numpy.reshape(f, (-1, f.shape[3]))
        grams.append(numpy.matmul(f.T, f) / f.size)

    return grams

