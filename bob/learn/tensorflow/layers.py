import math
import numbers

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAvgPool2D


def _check_input(
    value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(name)
            )
        value = [center - float(value), center + float(value)]
        if clip_first_on_zero:
            value[0] = max(value[0], 0.0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name, bound))
    else:
        raise TypeError(
            "{} should be a single number or a list/tuple with lenght 2.".format(name)
        )

    # # if value is 0 or (1., 1.) for brightness/contrast/saturation
    # # or (0., 0.) for hue, do nothing
    # if value[0] == value[1] == center:
    #     value = None
    return value


class ColorJitter(tf.keras.layers.Layer):
    """Adjust the brightness, contrast, saturation, and hue of an image or images by a random factor.

    Equivalent to adjust_brightness() using a delta randomly picked in the interval [-max_delta, max_delta)

    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * brightness_factor + mean`.

    Input shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.

    Output shape:
        4D tensor with shape:
        `(samples, height, width, channels)`, data_format='channels_last'.

    Attributes:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        seed (int or None): Used to create a random seed.
        name (str or None): The name of the layer.

    Raise:
        ValueError: if lower bound is not between [0, 1], or upper bound is negative.
    """

    def __init__(
        self,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        seed=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.brightness = _check_input(brightness, "brightness")
        self.contrast = _check_input(contrast, "contrast")
        self.saturation = _check_input(saturation, "saturation")
        self.hue = _check_input(
            hue, "hue", center=0, bound=(-1.0, 1.0), clip_first_on_zero=False
        )
        self.seed = seed
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    @tf.function
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        output = inputs
        if training:
            fn_idx = tf.convert_to_tensor(list(range(4)))
            fn_idx = tf.random.shuffle(fn_idx, seed=self.seed)
            if inputs.dtype not in (tf.dtypes.float16, tf.dtypes.float32):
                output = tf.image.convert_image_dtype(output, dtype=tf.dtypes.float32)

            for fn_id in fn_idx:
                if fn_id == 0:
                    brightness_factor = tf.random.uniform(
                        [],
                        minval=self.brightness[0],
                        maxval=self.brightness[1],
                        seed=self.seed,
                    )
                    output = tf.image.adjust_brightness(output, brightness_factor)

                if fn_id == 1:
                    contrast_factor = tf.random.uniform(
                        [],
                        minval=self.contrast[0],
                        maxval=self.contrast[1],
                        seed=self.seed,
                    )
                    output = tf.image.adjust_contrast(output, contrast_factor)

                if fn_id == 2:
                    saturation_factor = tf.random.uniform(
                        [],
                        minval=self.saturation[0],
                        maxval=self.saturation[1],
                        seed=self.seed,
                    )
                    output = tf.image.adjust_saturation(output, saturation_factor)

                if fn_id == 3:
                    hue_factor = tf.random.uniform(
                        [], minval=self.hue[0], maxval=self.hue[1], seed=self.seed
                    )
                    output = tf.image.adjust_hue(output, hue_factor)

            output = tf.image.convert_image_dtype(
                output, dtype=inputs.dtype, saturate=True
            )

        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def Normalize(mean, std=1.0, **kwargs):
    scale = 1.0 / std
    offset = mean / std
    return tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=scale, offset=offset, **kwargs
    )


class SphereFaceLayer(tf.keras.layers.Layer):
    r"""
    Implements the SphereFace loss from equation (7) of `SphereFace: Deep Hypersphere Embedding for Face Recognition <https://arxiv.org/abs/1704.08063>`_

    If the parameter `original` is set to `True` it will computes exactly what's written in eq (7): :math:`\\text{soft}(x_i) = \\frac{exp(||x_i||\\text{cos}(\\psi(\\theta_{yi})))}{exp(||x_i||\\text{cos}(\\psi(\\theta_{yi}))) + \sum_{j;j\\neq yi}  exp(||x_i||\\text{cos}(\psi(\\theta_{j}))) }`.
    Where :math:`\\psi(\\theta) = -1^k \\text{cos}(m\\theta)-2k`.

    Parameters
    ----------

      n_classes: int
        Number of classes

      m: float
         Margin

    """

    def __init__(self, n_classes=10, m=0.5):
        super(SphereFaceLayer, self).__init__(name="sphere_face_logits")
        self.n_classes = n_classes
        self.m = m

    def build(self, input_shape):
        super(SphereFaceLayer, self).build(input_shape[0])
        shape = [input_shape[-1], self.n_classes]

        self.W = self.add_variable("W", shape=shape)
        self.pi = tf.constant(math.pi)

    def call(self, X, training=None):

        # normalize feature
        X = tf.nn.l2_normalize(X, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # cos between X and W
        cos_yi = tf.matmul(X, W)
        cos_yi = tf.clip_by_value(cos_yi, -1, 1)

        # cos(m \theta)
        theta = tf.math.acos(cos_yi)
        cos_theta_m = tf.math.cos(self.m * theta)

        # ||x||
        x_norm = tf.norm(X, axis=-1, keepdims=True)

        # phi = -1**k * cos(m \theta) - 2k
        k = self.m * (theta / self.pi)
        phi = ((-(1 ** k)) * cos_theta_m) - 2 * k

        logits = x_norm * phi

        return logits


class ModifiedSoftMaxLayer(tf.keras.layers.Layer):
    """
    Implements the modified logit from equation (5) of `SphereFace: Deep Hypersphere Embedding for Face Recognition <https://arxiv.org/pdf/1704.08063.pdf>`_

    It basically transforms the regular logit function to :math:`||x_i||cos(\\theta_{yi})`, where :math:`\\theta_{yi}=||x_i||_2^2||W||_2^2`

    Parameters
    ----------

    n_classes: int
        Number of classes for the new logit function
    """

    def __init__(self, n_classes=10):

        super(ModifiedSoftMaxLayer, self).__init__(name="modified_softmax_logits")
        self.n_classes = n_classes

    def build(self, input_shape):
        super(ModifiedSoftMaxLayer, self).build(input_shape[0])
        shape = [input_shape[-1], self.n_classes]

        self.W = self.add_variable("W", shape=shape)

    def call(self, X, training=None):

        # normalize feature
        W = tf.nn.l2_normalize(self.W, axis=0)

        # cos between X and W
        cos_yi = tf.nn.l2_normalize(X, axis=1) @ W

        logits = tf.norm(X) * cos_yi

        return logits


def add_bottleneck(model, bottleneck_size=128, dropout_rate=0.2):
    """
    Amend a bottleneck layer to a Keras Model

    Parameters
    ----------

      model:
        Keras model

      bottleneck_size: int
         Size of the bottleneck

      dropout_rate: float
         Dropout rate
    """
    if not isinstance(model, tf.keras.models.Sequential):
        new_model = tf.keras.models.Sequential(model, name="bottleneck")
    else:
        new_model = model

    new_model.add(GlobalAvgPool2D())
    new_model.add(Dropout(dropout_rate, name="Dropout"))
    new_model.add(Dense(bottleneck_size, use_bias=False, name="embeddings"))
    new_model.add(BatchNormalization(axis=-1, scale=False, name="embeddings/BatchNorm"))

    return new_model


def add_top(model, n_classes):
    if not isinstance(model, tf.keras.models.Sequential):
        new_model = tf.keras.models.Sequential(model, name="logits")
    else:
        new_model = model

    new_model.add(Dense(n_classes, name="logits"))
    return new_model
