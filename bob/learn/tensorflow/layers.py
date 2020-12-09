import numbers

import tensorflow as tf


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
