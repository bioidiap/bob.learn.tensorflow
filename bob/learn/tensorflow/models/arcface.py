import math

import tensorflow as tf


class ArcFaceLayer(tf.keras.layers.Layer):
    """
    Implements the ArcFace from equation (3) of `ArcFace: Additive Angular Margin Loss for Deep Face Recognition <https://arxiv.org/abs/1801.07698>`_

    Defined as:

    :math:`s(cos(\\theta_i) + m`

    Parameters
    ----------

      n_classes: int
        Number of classes

      m: float
         Margin

      s: int
         Scale

      arc: bool
         If `True`, uses arcface loss. If `False`, it's a regular dense layer
    """

    def __init__(
        # don't forget to fix get_config when you change init params
        self,
        n_classes,
        s=30,
        m=0.5,
        arc=True,
        kernel_initializer=None,
        name="arc_face_logits",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.s = s
        self.arc = arc
        self.m = m
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        super(ArcFaceLayer, self).build(input_shape[0])
        shape = [input_shape[-1], self.n_classes]

        self.W = self.add_weight("W", shape=shape, initializer=self.kernel_initializer)

        self.cos_m = tf.identity(math.cos(self.m), name="cos_m")
        self.sin_m = tf.identity(math.sin(self.m), name="sin_m")
        self.th = tf.identity(math.cos(math.pi - self.m), name="th")
        self.mm = tf.identity(math.sin(math.pi - self.m) * self.m)

    def call(self, X, y, training=None):
        if self.arc:
            # normalize feature
            X = tf.nn.l2_normalize(X, axis=1)
            W = tf.nn.l2_normalize(self.W, axis=0)

            # cos between X and W
            cos_yi = tf.matmul(X, W)

            # sin_yi = tf.math.sqrt(1-cos_yi**2)
            sin_yi = tf.clip_by_value(tf.math.sqrt(1 - cos_yi ** 2), 0, 1)

            # cos(x+m) = cos(x)*cos(m) - sin(x)*sin(m)
            dtype = cos_yi.dtype
            cos_m = tf.cast(self.cos_m, dtype=dtype)
            sin_m = tf.cast(self.sin_m, dtype=dtype)
            th = tf.cast(self.th, dtype=dtype)
            mm = tf.cast(self.mm, dtype=dtype)

            cos_yi_m = cos_yi * cos_m - sin_yi * sin_m

            cos_yi_m = tf.where(cos_yi > th, cos_yi_m, cos_yi - mm)

            # Preparing the hot-output
            one_hot = tf.one_hot(
                tf.cast(y, tf.int32), depth=self.n_classes, name="one_hot_mask"
            )
            one_hot = tf.cast(one_hot, dtype=dtype)

            logits = (one_hot * cos_yi_m) + ((1.0 - one_hot) * cos_yi)
            logits = self.s * logits
        else:
            logits = tf.matmul(X, self.W)

        return logits

    def get_config(self):
        config = dict(super().get_config())
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "arc": self.arc,
                "m": self.m,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


class ArcFaceLayer3Penalties(tf.keras.layers.Layer):
    """
    Implements the ArcFace loss from equation (4) of `ArcFace: Additive Angular Margin Loss for Deep Face Recognition <https://arxiv.org/abs/1801.07698>`_

    Defined as:

      :math:`s(cos(m_1\\theta_i + m_2) -m_3`
    """

    def __init__(
        self,
        n_classes=10,
        s=30,
        m1=0.5,
        m2=0.5,
        m3=0.5,
        name="arc_face_logits",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def build(self, input_shape):
        super(ArcFaceLayer3Penalties, self).build(input_shape[0])
        shape = [input_shape[-1], self.n_classes]

        self.W = self.add_variable("W", shape=shape)

    def call(self, X, y, training=None):

        # normalize feature
        X = tf.nn.l2_normalize(X, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # cos between X and W
        cos_yi = tf.matmul(X, W)

        # Getting the angle
        theta = tf.math.acos(cos_yi)
        theta = tf.clip_by_value(
            theta, -1.0 + tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )

        cos_yi_m = tf.math.cos(self.m1 * theta + self.m2) - self.m3

        # logits = self.s*cos_theta_m

        # Preparing the hot-output
        one_hot = tf.one_hot(
            tf.cast(y, tf.int32), depth=self.n_classes, name="one_hot_mask"
        )

        one_hot = tf.cast(one_hot, cos_yi_m.dtype)

        logits = (one_hot * cos_yi_m) + ((1.0 - one_hot) * cos_yi)

        logits = self.s * logits
        return logits

    def get_config(self):
        config = dict(super().get_config())
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m1": self.m1,
                "m2": self.m2,
                "m3": self.m3,
            }
        )
        return config
