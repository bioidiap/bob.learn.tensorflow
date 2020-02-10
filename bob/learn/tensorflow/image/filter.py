import tensorflow as tf


def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution.
    Code adapted from: https://stackoverflow.com/a/52012658/1286165"""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum("i,j->ij", vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


class GaussianFilter:
    """A class for blurring images"""

    def __init__(self, size=13, mean=0.0, std=3.0, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.mean = mean
        self.std = std
        self.gauss_kernel = gaussian_kernel(size, mean, std)[:, :, None, None]

    def __call__(self, image):
        shape = tf.shape(image)
        image = tf.reshape(image, [-1, shape[-3], shape[-2], shape[-1]])
        input_channels = shape[-1]
        gauss_kernel = tf.tile(self.gauss_kernel, [1, 1, input_channels, 1])
        return tf.nn.depthwise_conv2d(
            image,
            gauss_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
        )
