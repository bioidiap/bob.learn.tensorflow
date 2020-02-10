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
from bob.learn.tensorflow.style_transfer import do_style_transfer


logger = logging.getLogger(__name__)


@click.command(
    entry_point_group='bob.learn.tensorflow.config', cls=ConfigCommand)
@click.argument('content_image_path', required=True)
@click.argument('output_path', required=True)
@click.option('--style-image-paths',
              cls=ResourceOption,
              required=True,
              multiple=True,
              entry_point_group='bob.learn.tensorflow.style_images',
              help='List of images that encodes the style.')
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
              help='CNN checkpoint path')
@click.option('--iterations',
              '-i',
              type=click.types.INT,
              help='Number of iterations to generate the image',
              default=1000)
@click.option('--learning_rate',
              '-r',
              type=click.types.FLOAT,
              help='Learning rate.',
              default=1.)
@click.option('--content-weight',
              type=click.types.FLOAT,
              help='Weight of the content loss.',
              default=5.)
@click.option('--style-weight',
              type=click.types.FLOAT,
              help='Weight of the style loss.',
              default=100.)
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
@click.option('--preprocess-fn',
              '-pr',
              cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.preprocess_fn',
              help='Preprocess function. Pointer to a function that preprocess the INPUT signal')
@click.option('--un-preprocess-fn',
              '-un',
              cls=ResourceOption,
              entry_point_group='bob.learn.tensorflow.preprocess_fn',
              help='Un preprocess function. Pointer to a function that preprocess the OUTPUT signal')
@click.option(
  '--start-from',
  '-sf',
  cls=ResourceOption,
  default="noise",
  type=click.Choice(["noise", "content", "style"]),
  help="Starts from this image for reconstruction",
)
@verbosity_option(cls=ResourceOption)
def style_transfer(content_image_path, output_path, style_image_paths,
                   architecture, checkpoint_dir,
                   iterations, learning_rate,
                   content_weight, style_weight, denoise_weight, content_end_points,
                   style_end_points, scopes, pure_noise, preprocess_fn,
                   un_preprocess_fn, start_from, **kwargs):
    """
     Trains neural style transfer using the approach presented in:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    \b

    If you want run a style transfer using InceptionV2 as basis, use the following template

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

    logger.info("Style transfer, content_image={0}, style_image={1}".format(content_image_path, style_image_paths))

    # Loading content image
    content_image = bob.io.base.load(content_image_path)

    # Reading and converting to the tensorflow format
    style_images = []
    for path in style_image_paths:
        style_images.append(bob.io.base.load(path))

    output = do_style_transfer(content_image, style_images,
                               architecture, checkpoint_dir, scopes,
                               content_end_points, style_end_points,
                               preprocess_fn=preprocess_fn, un_preprocess_fn=un_preprocess_fn,
                               pure_noise=pure_noise,
                               iterations=iterations, learning_rate=learning_rate,
                               content_weight=content_weight, style_weight=style_weight,
                               denoise_weight=denoise_weight, start_from=start_from)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bob.io.base.save(output, output_path)
