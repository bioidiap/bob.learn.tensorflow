import tensorflow as tf


def gram_matrix(input_tensor):
    """Computes the gram matrix

    Parameters
    ----------
    input_tensor : object
        The input tensor. Usually it's the activation of a conv layer. The input shape
        must be ``BHWC``.

    Returns
    -------
    object
        The computed gram matrix as a tensor.

    Example
    -------
    >>>> gram_matrix(tf.zeros((32, 4, 6, 12)))
    <tf.Tensor: id=53, shape=(32, 12, 12), dtype=float32, numpy=
    array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def upper_triangle_and_diagonal(A):
    """Returns a flat version of upper triangle of a 2D array (including diagonal).

    This function is useful to be applied on gram matrices since they contain duplicate
    information.

    Parameters
    ----------
    A
        A two dimensional array.

    Returns
    -------
    object
        The flattened upper triangle of array

    Example
    -------
    >>> A = [
    ...  [1, 2, 3],
    ...  [4, 5, 6],
    ...  [7, 8, 9],
    ... ]
    >>> upper_triangle_and_diagonal(A)
    [1,2,3,5,6,9]
    """
    ones = tf.ones_like(A)
    # Upper triangular matrix of 0s and 1s (including diagonal)
    mask = tf.matrix_band_part(ones, 0, -1)
    upper_triangular_flat = tf.boolean_mask(A, mask)
    return upper_triangular_flat


def upper_triangle(A):
    ones = tf.ones_like(A)
    # Upper triangular matrix of 0s and 1s (including diagonal)
    mask_a = tf.matrix_band_part(ones, 0, -1)
    # Diagonal
    mask_b = tf.matrix_band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)
    upper_triangular_flat = tf.boolean_mask(A, mask)
    return upper_triangular_flat
