#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import logging
import tensorflow as tf
import functools
logger = logging.getLogger(__name__)


def content_loss(noises, content_features):
    """

    Implements the content loss from:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    For a given noise signal :math:`n`, content image :math:`c` and convolved with the DCNN :math:`\phi` until the layer :math:`l` the content loss is defined as:

    :math:`L(n,c) = \sum_{l=?}^{?}({\phi^l(n) - \phi^l(c)})^2`


    Parameters
    ----------

     noises: :any:`list`
        A list of tf.Tensor containing all the noises convolved

     content_features: :any:`list`
        A list of numpy.array containing all the content_features convolved

    """

    content_losses = []
    for n,c in zip(noises, content_features):
        content_losses.append((2 * tf.nn.l2_loss(n - c) / c.size))
    return functools.reduce(tf.add, content_losses)


def linear_gram_style_loss(noises, gram_style_features):
    """

    Implements the style loss from:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    For a given noise signal :math:`n`, content image :math:`c` and convolved with the DCNN :math:`\phi` until the layer :math:`l` the STYLE loss is defined as

    :math:`L(n,c) = \\sum_{l=?}^{?}\\frac{({\phi^l(n)^T*\\phi^l(n) - \\phi^l(c)^T*\\phi^l(c)})^2}{N*M}`


    Parameters
    ----------

     noises: :any:`list`
        A list of tf.Tensor containing all the noises convolved

     gram_style_features: :any:`list`
        A list of numpy.array containing all the content_features convolved

    """

    style_losses = []
    for n,s in zip(noises, gram_style_features):
        style_losses.append((2 * tf.nn.l2_loss(n - s)) / s.size)

    return functools.reduce(tf.add, style_losses)



def denoising_loss(noise):
    """
    Computes the denoising loss as in:

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).

    Parameters
    ----------

       noise:
          Input noise

    """
    def _tensor_size(tensor):
        from operator import mul
        return functools.reduce(mul, (d.value for d in tensor.get_shape()), 1)

    shape = noise.get_shape().as_list()

    noise_y_size = _tensor_size(noise[:,1:,:,:])
    noise_x_size = _tensor_size(noise[:,:,1:,:])
    denoise_loss = 2 * ( (tf.nn.l2_loss(noise[:,1:,:,:] - noise[:,:shape[1]-1,:,:]) / noise_y_size) +
                    (tf.nn.l2_loss(noise[:,:,1:,:] - noise[:,:,:shape[2]-1,:]) / noise_x_size))

    return denoise_loss

