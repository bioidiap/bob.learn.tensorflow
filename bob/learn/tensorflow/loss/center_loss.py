import tensorflow as tf

# TODO(amir): replace parent class with tf.Module in tensorflow 1.14 and above.
# * pass ``name`` to parent class
# * replace get_variable with tf.Variable
# * replace variable_scope with name_scope
class CenterLoss:
    """Center loss."""

    def __init__(self, n_classes, n_features, alpha=0.9, name="center_loss", **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha
        self.name = name
        with tf.variable_scope(self.name):
            self.centers = tf.get_variable(
                "centers",
                [n_classes, n_features],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.),
                trainable=False,
            )

    def __call__(self, sparse_labels, prelogits):
        with tf.name_scope(self.name):
            centers_batch = tf.gather(self.centers, sparse_labels)
            diff = (1 - self.alpha) * (centers_batch - prelogits)
            self.centers_update_op = tf.scatter_sub(self.centers, sparse_labels, diff)
            center_loss = tf.reduce_mean(tf.square(prelogits - centers_batch))
        tf.summary.scalar("loss_center", center_loss)
        # Add histogram for all centers
        for i in range(self.n_classes):
            tf.summary.histogram(f"center_{i}", self.centers[i])
        return center_loss

    @property
    def update_ops(self):
        return [self.centers_update_op]
