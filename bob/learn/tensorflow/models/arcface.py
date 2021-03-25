import math

import tensorflow as tf

from bob.learn.tensorflow.metrics.embedding_accuracy import accuracy_from_embeddings

from .embedding_validation import EmbeddingValidation


class ArcFaceModel(EmbeddingValidation):
    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:

            logits, _ = self((X, y), training=True)
            loss = self.compiled_loss(
                y, logits, sample_weight=None, regularization_losses=self.losses
            )
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, logits, sample_weight=None)

        self.train_loss(loss)
        return {m.name: m.result() for m in self.metrics + [self.train_loss]}

    def test_step(self, data):
        """
        Test Step
        """

        images, labels = data

        # No worries, labels not used in validation
        _, embeddings = self((images, labels), training=False)
        self.validation_acc(accuracy_from_embeddings(labels, embeddings))
        return {m.name: m.result() for m in [self.validation_acc]}


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
    """

    def __init__(self, n_classes=10, s=30, m=0.5):
        super(ArcFaceLayer, self).__init__(name="arc_face_logits")
        self.n_classes = n_classes
        self.s = s
        self.m = m

    def build(self, input_shape):
        super(ArcFaceLayer, self).build(input_shape[0])
        shape = [input_shape[-1], self.n_classes]

        self.W = self.add_variable("W", shape=shape)

        self.cos_m = tf.identity(math.cos(self.m), name="cos_m")
        self.sin_m = tf.identity(math.sin(self.m), name="sin_m")
        self.th = tf.identity(math.cos(math.pi - self.m), name="th")
        self.mm = tf.identity(math.sin(math.pi - self.m) * self.m)

    def call(self, X, y, training=None):

        # normalize feature
        X = tf.nn.l2_normalize(X, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # cos between X and W
        cos_yi = tf.matmul(X, W)

        # sin_yi = tf.math.sqrt(1-cos_yi**2)
        sin_yi = tf.clip_by_value(tf.math.sqrt(1 - cos_yi ** 2), 0, 1)

        # cos(x+m) = cos(x)*cos(m) - sin(x)*sin(m)
        cos_yi_m = cos_yi * self.cos_m - sin_yi * self.sin_m

        cos_yi_m = tf.where(cos_yi > self.th, cos_yi_m, cos_yi - self.mm)

        # Preparing the hot-output
        one_hot = tf.one_hot(
            tf.cast(y, tf.int32), depth=self.n_classes, name="one_hot_mask"
        )

        logits = (one_hot * cos_yi_m) + ((1.0 - one_hot) * cos_yi)
        logits = self.s * logits

        return logits


class ArcFaceLayer3Penalties(tf.keras.layers.Layer):
    """
    Implements the ArcFace loss from equation (4) of `ArcFace: Additive Angular Margin Loss for Deep Face Recognition <https://arxiv.org/abs/1801.07698>`_

    Defined as:

      :math:`s(cos(m_1\\theta_i + m_2) -m_3`
    """

    def __init__(self, n_classes=10, s=30, m1=0.5, m2=0.5, m3=0.5):
        super(ArcFaceLayer3Penalties, self).__init__(name="arc_face_logits")
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

        cos_yi_m = tf.math.cos(self.m1 * theta + self.m2) - self.m3

        # logits = self.s*cos_theta_m

        # Preparing the hot-output
        one_hot = tf.one_hot(
            tf.cast(y, tf.int32), depth=self.n_classes, name="one_hot_mask"
        )

        logits = (one_hot * cos_yi_m) + ((1.0 - one_hot) * cos_yi)

        logits = self.s * logits

        return logits
