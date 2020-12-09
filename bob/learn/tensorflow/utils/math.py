import tensorflow as tf


def gram_matrix(input_tensor):
    """Computes the gram matrix.

    Parameters
    ----------
    input_tensor
        The input tensor. Usually it's the activation of a conv layer. The input
        shape must be ``BHWC``.

    Returns
    -------
    object
        The computed gram matrix as a tensor.

    Example
    -------
    >>> from bob.learn.tensorflow.utils import gram_matrix
    >>> gram_matrix(tf.zeros((32, 4, 6, 12))).numpy().shape
    (32, 12, 12)
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input=input_tensor)
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
    >>> from bob.learn.tensorflow.utils import upper_triangle_and_diagonal
    >>> A = [
    ...  [1, 2, 3],
    ...  [4, 5, 6],
    ...  [7, 8, 9],
    ... ]
    >>> upper_triangle_and_diagonal(A).numpy()
    array([1, 2, 3, 5, 6, 9], dtype=int32)
    """
    ones = tf.ones_like(A)
    # Upper triangular matrix of 0s and 1s (including diagonal)
    mask = tf.linalg.band_part(ones, 0, -1)
    upper_triangular_flat = tf.boolean_mask(tensor=A, mask=mask)
    return upper_triangular_flat


def upper_triangle(A):
    ones = tf.ones_like(A)
    # Upper triangular matrix of 0s and 1s (including diagonal)
    mask_a = tf.linalg.band_part(ones, 0, -1)
    # Diagonal
    mask_b = tf.linalg.band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)
    upper_triangular_flat = tf.boolean_mask(tensor=A, mask=mask)
    return upper_triangular_flat


def pdist(A, metric="sqeuclidean"):
    if metric != "sqeuclidean":
        raise NotImplementedError()
    r = tf.reduce_sum(input_tensor=A * A, axis=1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(A, A, transpose_b=True) + tf.transpose(a=r)
    return D


def cdist(A, B, metric="sqeuclidean"):
    if metric != "sqeuclidean":
        raise NotImplementedError()
    M1, M2 = tf.shape(input=A)[0], tf.shape(input=B)[0]
    # code from https://stackoverflow.com/a/43839605/1286165
    p1 = tf.matmul(
        tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(A), axis=1), 1),
        tf.ones(shape=(1, M2)),
    )
    p2 = tf.transpose(
        a=tf.matmul(
            tf.reshape(tf.reduce_sum(input_tensor=tf.square(B), axis=1), shape=[-1, 1]),
            tf.ones(shape=(M1, 1)),
            transpose_b=True,
        )
    )

    D = tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)
    return D


def random_choice_no_replacement(one_dim_input, num_indices_to_drop=3, sort=False):
    """Similar to np.random.choice with no replacement.
    Code from https://stackoverflow.com/a/54755281/1286165
    """
    input_length = tf.shape(input=one_dim_input)[0]

    # create uniform distribution over the sequence
    uniform_distribution = tf.random.uniform(
        shape=[input_length],
        minval=0,
        maxval=None,
        dtype=tf.float32,
        seed=None,
        name=None,
    )

    # grab the indices of the greatest num_words_to_drop values from the distibution
    _, indices_to_keep = tf.nn.top_k(
        uniform_distribution, input_length - num_indices_to_drop
    )

    # sort the indices
    if sort:
        sorted_indices_to_keep = tf.sort(indices_to_keep)
    else:
        sorted_indices_to_keep = indices_to_keep

    # gather indices from the input array using the filtered actual array
    result = tf.gather(one_dim_input, sorted_indices_to_keep)
    return result
