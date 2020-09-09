import tensorflow as tf


class CenterLoss(tf.keras.losses.Loss):
    """Center loss."""

    def __init__(self, n_classes, n_features, alpha=0.9, name="center_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha

        self.centers = tf.Variable(
            tf.zeros([n_classes, n_features]), name="centers", trainable=False
        )

    def call(self, y_true, y_pred):
        sparse_labels, prelogits = tf.reshape(y_true, (-1,)), y_pred
        centers_batch = tf.gather(self.centers, sparse_labels)
        diff = (1 - self.alpha) * (centers_batch - prelogits)
        center_loss = tf.reduce_mean(input_tensor=tf.square(prelogits - centers_batch))
        self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, sparse_labels[:, None], diff))
        return center_loss
