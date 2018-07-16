#!/usr/bin/env python
"""Trains networks using Tensorflow estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import click
import tensorflow as tf
from bob.extension.scripts.click_helper import (verbosity_option,
                                                ConfigCommand, ResourceOption)
import bob.io.image
import bob.io.base
import numpy
import bob.ip.base
import bob.ip.color
import sys
import os
from bob.learn.tensorflow.style_transfer import compute_features, compute_gram
from bob.learn.tensorflow.loss import linear_gram_style_loss, content_loss, denoising_loss


logger = logging.getLogger(__name__)

def wise_shape(shape):
    if len(shape)==2:
        return (1, shape[0], shape[1], 1)
    else:
        return (1, shape[0], shape[1], shape[2])

def normalize4save(img):
    return (255 * ((img - numpy.min(img)) / (numpy.max(img)-numpy.min(img)))).astype("uint8")


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.argument('content_image_path', required=True)
@click.argument('output_path', required=True)
@click.option('--style-image-paths',
              cls=ResourceOption,
              required=True,
              multiple=True,              
              entry_point_group='bob.learn.tensorflow.style_images',
              help='List of images that encods the style.')
@click.option('--architecture',
              '-a',
              required=True,
              cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.architecture',
              help='The base architecure.')
@click.option('--checkpoint-dir',
              '-c',
              required=True,
              cls=ResourceOption,
              help='The base architecure.')
@click.option('--iterations',
              '-i',
              type=click.types.INT,
              help='Number of steps for which to train model.',
              default=1000)
@click.option('--learning_rate',
              '-i',
              type=click.types.FLOAT,
              help='Learning rate.',
              default=1.)
@click.option('--content-weight',
              type=click.types.FLOAT,
              help='Weight of the content loss.',
              default=1.)
@click.option('--style-weight',
              type=click.types.FLOAT,
              help='Weight of the style loss.',
              default=1000.)
@click.option('--denoise-weight',
              type=click.types.FLOAT,
              help='Weight denoising loss.',
              default=100.)
@click.option('--content-end-points',
              cls=ResourceOption,
              multiple=True,
              entry_point_group='bob.learn.tensorflow.end_points',
              help='List of end_points for the used to encode the content')
@click.option('--style-end-points',
              cls=ResourceOption,
              multiple=True,
              entry_point_group='bob.learn.tensorflow.end_points',
              help='List of end_points for the used to encode the style')
@click.option('--scopes',
              cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.scopes',
              help='Dictionary containing the mapping scores',
              required=True)
@click.option('--pure-noise',
               is_flag=True,
               help="If set will save the raw noisy generated image."
                    "If not set, the output will be RGB = stylizedYUV.Y, originalYUV.U, originalYUV.V"
              )
@verbosity_option(cls=ResourceOption)
def style_transfer(content_image_path, output_path, style_image_paths,
                   architecture, checkpoint_dir,
                   iterations, learning_rate,
                   content_weight, style_weight, denoise_weight, content_end_points,
                   style_end_points, scopes, pure_noise,  **kwargs):
    """
     Trains neural style transfer using the approach presented in:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    \b

    If you want run a style transfer using InceptionV2 as basis folo

    Below follow a CONFIG template
    
    CONFIG.PY
    ```

       from bob.extension import rc

       from bob.learn.tensorflow.network import inception_resnet_v2_batch_norm
       architecture = inception_resnet_v2_batch_norm

       checkpoint_dir = rc["bob.bio.face_ongoing.idiap_casia_inception_v2_centerloss_rgb"]

       style_end_points = ["Conv2d_1a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3"]

       content_end_points = ["Bottleneck", "PreLogitsFlatten"]

       scopes = {"InceptionResnetV2/":"InceptionResnetV2/"}

    ```
    \b

    Then run::

       $ bob tf style <content-image> <output-image> --style-image-paths <style-image> CONFIG.py


    You can also provide a list of images to encode the style using the config file as in the example below.

    CONFIG.PY
    ```

       from bob.extension import rc

       from bob.learn.tensorflow.network import inception_resnet_v2_batch_norm
       architecture = inception_resnet_v2_batch_norm

       checkpoint_dir = rc["bob.bio.face_ongoing.idiap_casia_inception_v2_centerloss_rgb"]

       style_end_points = ["Conv2d_1a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3"]

       content_end_points = ["Bottleneck", "PreLogitsFlatten"]

       scopes = {"InceptionResnetV2/":"InceptionResnetV2/"}

       style_image_paths = ["STYLE_1.png",
                            "STYLE_2.png"]

    ```
 
    Then run::

       $ bob tf style <content-image> <output-image> CONFIG.py

    \b \b

    """

    # Reading and converting to the tensorflow format
    content_image = bob.io.image.to_matplotlib(bob.io.base.load(content_image_path)).astype("float32")
    style_images = []
    for path in style_image_paths:
        image = bob.io.image.to_matplotlib(bob.io.base.load(path)).astype("float32")
        style_images.append(numpy.reshape(image, wise_shape(image.shape)))

    # Reshaping to NxWxHxC
    content_image = numpy.reshape(content_image, wise_shape(content_image.shape))

    # Base content features
    logger.info("Computing content features")
    content_features = compute_features(content_image, architecture, checkpoint_dir, content_end_points)

    # Base style features
    logger.info("Computing style features")  
    style_grams = []
    for image in style_images:
        style_features = compute_features(image, architecture, checkpoint_dir, style_end_points)
        style_grams.append(compute_gram(style_features))

    # Organizing the trainer
    logger.info("Training.....")
    with tf.Graph().as_default():
        tf.set_random_seed(0)

        # Random noise
        noise = tf.Variable(tf.random_normal(shape=content_image.shape),
                            trainable=True)

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
                sys.stdout.flush()

            # Saving generated image
            raw_style_image = sess.run(noise)[0, :, :,:]
            raw_style_image = bob.io.image.to_bob(raw_style_image)
            normalized_style_image = normalize4save(raw_style_image)

            if pure_noise:
                if normalized_style_image.shape[0] == 1:
                    bob.io.base.save(normalized_style_image[0, :, :], output_path)
                else:
                    bob.io.base.save(normalized_style_image, output_path)
            else:
                # Original output
                if normalized_style_image.shape[0] == 1:
                    normalized_style_image_yuv = bob.ip.color.rgb_to_yuv(bob.ip.color.gray_to_rgb(normalized_style_image[0,:,:]))
                    # Loading the content image and clipping from 0-255 in case is in another scale
                    scaled_content_image = normalize4save(bob.io.base.load(content_image_path).astype("float32")).astype("float64")
                    content_image_yuv = bob.ip.color.rgb_to_yuv(bob.ip.color.gray_to_rgb(scaled_content_image))
                else:
                    normalized_style_image_yuv = bob.ip.color.rgb_to_yuv(bob.ip.color.gray_to_rgb(bob.ip.color.rgb_to_gray(normalized_style_image)))
                    content_image_yuv = bob.ip.color.rgb_to_yuv(bob.io.base.load(content_image_path))
                
                output_image = numpy.zeros(shape=content_image_yuv.shape, dtype="uint8")
                output_image[0,:,:] = normalized_style_image_yuv[0,:,:]
                output_image[1,:,:] = content_image_yuv[1,:,:]
                output_image[2,:,:] = content_image_yuv[2,:,:]

                output_image = bob.ip.color.yuv_to_rgb(output_image)
                bob.io.base.save(output_image, output_path)

