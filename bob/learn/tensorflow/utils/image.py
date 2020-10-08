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
    ndim = image.ndim
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
    ndim = image.ndim
    if ndim < 3:
        raise ValueError(
            "The image needs to be at least 3 dimensional but it " "was {}".format(ndim)
        )
    axis_order = [2, 0, 1]
    shift = ndim - 3
    axis_order = list(range(ndim - 3)) + [n + shift for n in axis_order]
    return tf.transpose(a=image, perm=axis_order)
