import numpy as np
import tensorflow as tf


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
    return tf.transpose(a=image, perm=axis_order)


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
    return tf.transpose(a=image, perm=axis_order)


def blocks_tensorflow(images, block_size):
    """Return all non-overlapping blocks of an image using tensorflow
    operations.

    Parameters
    ----------
    images : `tf.Tensor`
        The input color images. It is assumed that the image has a shape of
        [?, H, W, C].
    block_size : (int, int)
        A tuple of two integers indicating the block size.

    Returns
    -------
    blocks : `tf.Tensor`
        All the blocks in the batch dimension. The output will be of
        size [?, block_size[0], block_size[1], C].
    n_blocks : int
        The number of blocks that was obtained per image.
    """
    # normalize block_size
    block_size = [1] + list(block_size) + [1]
    output_size = list(block_size)
    output_size[0] = -1
    output_size[-1] = images.shape[-1]
    blocks = tf.image.extract_patches(
        images, block_size, block_size, [1, 1, 1, 1], "VALID"
    )
    n_blocks = int(np.prod(blocks.shape[1:3]))
    output = tf.reshape(blocks, output_size)
    return output, n_blocks


def tf_repeat(tensor, repeats):
    """
    Parameters
    ----------
    tensor
        A Tensor. 1-D or higher.
    repeats
        A list. Number of repeat for each dimension, length must be the same as
        the number of dimensions in input

    Returns
    -------
    A Tensor. Has the same type as input. Has the shape of tensor.shape *
    repeats
    """
    with tf.name_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(input=tensor) * repeats)
    return repeated_tesnor


def all_patches(image, label, key, size):
    """Extracts all patches of an image

    Parameters
    ----------
    image:
        The image should be channels_last format and already batched.

    label:
        The label for the image

    key:
        The key for the image

    size: (int, int)
        The height and width of the blocks.

    Returns
    -------
    blocks:
       The non-overlapping blocks of size from image and labels and keys are
       repeated.

    label:

    key:
    """
    blocks, n_blocks = blocks_tensorflow(image, size)

    # duplicate label and key as n_blocks
    def repeats(shape):
        r = list(shape)
        for i in range(len(r)):
            if i == 0:
                r[i] = n_blocks
            else:
                r[i] = 1
        return r

    label = tf_repeat(label, repeats(label.shape))
    key = tf_repeat(key, repeats(key.shape))

    return blocks, label, key
