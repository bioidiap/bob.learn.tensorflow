import tensorflow as tf

from bob.learn.tensorflow.metrics.embedding_accuracy import accuracy_from_embeddings


class EmbeddingValidation(tf.keras.Model):
    """
    Use this model if the validation step should validate the accuracy with respect to embeddings.

    In this model, the `test_step` runs the function `bob.learn.tensorflow.metrics.embedding_accuracy.accuracy_from_embeddings`
    """

    def compile(
        self,
        **kwargs,
    ):
        """
        Compile
        """
        super().compile(**kwargs)
        self.train_loss = tf.keras.metrics.Mean(name="accuracy")
        self.validation_acc = tf.keras.metrics.Mean(name="accuracy")

    def train_step(self, data):
        """
        Train Step
        """

        X, y = data
        with tf.GradientTape() as tape:
            logits, _ = self(X, training=True)
            loss = self.loss(y, logits)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, logits, sample_weight=None)
        self.train_loss(loss)
        return {m.name: m.result() for m in self.metrics + [self.train_loss]}

        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.train_loss(loss)
        # return {m.name: m.result() for m in [self.train_loss]}

    def test_step(self, data):
        """
        Test Step
        """

        images, labels = data
        logits, prelogits = self(images, training=False)
        self.validation_acc(accuracy_from_embeddings(labels, prelogits))
        return {m.name: m.result() for m in [self.validation_acc]}
