#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


import tensorflow as tf
import numpy
import os
from bob.learn.tensorflow.loss import linear_gram_style_loss, content_loss, denoising_loss
import bob.io.image
import bob.ip.color

import logging
logger = logging.getLogger(__name__)


def compute_features(input_image, architecture, checkpoint_dir, target_end_points, preprocess_fn=None):
    """
    For a given set of end_points, convolve the input image until these points

    Parameters
    ----------

    input_image: :any:`numpy.array`
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
        features = sess.run([end_points[ep] for ep in target_end_points], feed_dict={input_pl: input_image})

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

    features: :any:`numpy.array`
      Convolved features in the format NxWxHxC

    """

    grams = []
    for f in features:
        f = numpy.reshape(f, (-1, f.shape[3]))
        grams.append(numpy.matmul(f.T, f) / f.size)

    return grams


def do_style_transfer(content_image, style_images,
                      architecture, checkpoint_dir, scopes,
                      content_end_points, style_end_points,
                      preprocess_fn=None, un_preprocess_fn=None, pure_noise=False,
                      iterations=1000, learning_rate=0.1,
                      content_weight=5., style_weight=500., denoise_weight=500., start_from="noise"):

    """
    Trains neural style transfer using the approach presented in:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    Parameters
    ----------

    content_image: :any:`numpy.array`
       Content image in the Bob format (C x W x H)

    style_images: :any:`list`
       List of numpy.array (Bob format (C x W x H)) that encodes the style

    architecture:
       Point to a function with the base architecture

    checkpoint_dir:
       CNN checkpoint path

    scopes:
       Dictionary containing the mapping scores

    content_end_points:
       List of end_points (from the architecture) for the used to encode the content

    style_end_points:
       List of end_points (from the architecture) for the used to encode the style

    preprocess_fn:
       Preprocess function. Pointer to a function that preprocess the INPUT signal

    unpreprocess_fn:
       Un preprocess function. Pointer to a function that preprocess the OUTPUT signal

    pure_noise:
       If set will save the raw noisy generated image.
       If not set, the output will be RGB = stylizedYUV.Y, originalYUV.U, originalYUV.V

    iterations:
       Number of iterations to generate the image

    learning_rate:
       Adam learning rate

    content_weight:
       Weight of the content loss

    style_weight:
       Weight of the style loss

    denoise_weight:
       Weight denoising loss
    """

    def wise_shape(shape):
        if len(shape)==2:
            return (1, shape[0], shape[1], 1)
        else:
            return (1, shape[0], shape[1], shape[2])

    def normalize4save(img):
        return (255 * ((img - numpy.min(img)) / (numpy.max(img)-numpy.min(img)))).astype("uint8")

    # Reshaping to NxWxHxC and converting to the tensorflow format
    # content
    original_image = content_image
    content_image = bob.io.image.to_matplotlib(content_image).astype("float32")
    content_image = numpy.reshape(content_image, wise_shape(content_image.shape))

    # and style
    for i in range(len(style_images)):
        image = bob.io.image.to_matplotlib(style_images[i])
        image = numpy.reshape(image, wise_shape(image.shape))
        style_images[i] = image

    # Base content features
    logger.info("Computing content features")
    content_features = compute_features(content_image, architecture, checkpoint_dir,
                                        content_end_points, preprocess_fn)

    # Base style features
    logger.info("Computing style features")
    style_grams = []
    for image in style_images:
        style_features = compute_features(image, architecture, checkpoint_dir,
                                          style_end_points, preprocess_fn)
        style_grams.append(compute_gram(style_features))

    # Organizing the trainer
    logger.info("Training.....")
    with tf.Graph().as_default():
        tf.set_random_seed(0)

        # Random noise
        if start_from == "noise":
            starting_image = tf.random_normal(shape=content_image.shape) * 0.256
        elif start_from == "content":
            starting_image = preprocess_fn(content_image)
        elif start_from == "style":
            starting_image = preprocess_fn(style_images[0])
        else:
            raise ValueError(f"Unknown starting image: {start_from}")

        noise = tf.Variable(starting_image, dtype="float32", trainable=True)
        _, end_points = architecture(noise,
                                      mode=tf.estimator.ModeKeys.PREDICT,
                                      trainable_variables=[])

        # Computing content loss
        content_noises = []
        for c in content_end_points:
            content_noises.append(end_points[c])
        c_loss = content_loss(content_noises, content_features)

        # Computing style_loss
        style_gram_noises = []
        s_loss = 0

        for grams_per_image in style_grams:

            for c in style_end_points:
                layer = end_points[c]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                features = tf.reshape(layer, (-1, number))
                style_gram_noises.append(tf.matmul(tf.transpose(features), features) / size)
            s_loss += linear_gram_style_loss(style_gram_noises, grams_per_image)

        # Variation denoise
        d_loss = denoising_loss(noise)

        #Total loss
        total_loss = content_weight*c_loss + style_weight*s_loss + denoise_weight*d_loss

        solver = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        tf.contrib.framework.init_from_checkpoint(tf.train.latest_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else checkpoint_dir, scopes)
        # Training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(iterations):
                _, loss = sess.run([solver, total_loss])
                logger.info("Iteration {0}, loss {1}".format(i, loss))

            # Saving generated image
            raw_style_image = sess.run(noise)[0, :, :,:]
            # Unpreprocessing the signal
            if un_preprocess_fn is not None:
                raw_style_image = un_preprocess_fn(raw_style_image)

            raw_style_image = bob.io.image.to_bob(raw_style_image)
            normalized_style_image = normalize4save(raw_style_image)

            if pure_noise:
                if normalized_style_image.shape[0] == 1:
                    return normalized_style_image[0, :, :]
                else:
                    return normalized_style_image
            else:
                # Original output
                if normalized_style_image.shape[0] == 1:
                    normalized_style_image_yuv = bob.ip.color.rgb_to_yuv(bob.ip.color.gray_to_rgb(normalized_style_image[0,:,:]))
                    # Loading the content image and clipping from 0-255 in case is in another scale
                    #scaled_content_image = normalize4save(bob.io.base.load(content_image_path).astype("float32")).astype("float64")
                    scaled_content_image = original_image.astype("float64")
                    content_image_yuv = bob.ip.color.rgb_to_yuv(bob.ip.color.gray_to_rgb(scaled_content_image))
                else:
                    normalized_style_image_yuv = bob.ip.color.rgb_to_yuv(bob.ip.color.gray_to_rgb(bob.ip.color.rgb_to_gray(normalized_style_image)))
                    content_image_yuv = bob.ip.color.rgb_to_yuv(original_image)

                output_image = numpy.zeros(shape=content_image_yuv.shape, dtype="uint8")
                output_image[0,:,:] = normalized_style_image_yuv[0,:,:]
                output_image[1,:,:] = content_image_yuv[1,:,:]
                output_image[2,:,:] = content_image_yuv[2,:,:]

                output_image = bob.ip.color.yuv_to_rgb(output_image)
                return output_image

