import tensorflow as tf


def compute_kernel(x, y):
    """Gaussian kernel.
    """
    x_size = tf.shape(input=x)[0]
    y_size = tf.shape(input=y)[0]
    dim = tf.shape(input=x)[1]
    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.exp(
        -tf.reduce_mean(input_tensor=tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32)
    )


def mmd(x, y):
    """Maximum Mean Discrepancy with Gaussian kernel.
    See: https://stats.stackexchange.com/a/276618/49433
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return (
        tf.reduce_mean(input_tensor=x_kernel)
        + tf.reduce_mean(input_tensor=y_kernel)
        - 2 * tf.reduce_mean(input_tensor=xy_kernel)
    )
