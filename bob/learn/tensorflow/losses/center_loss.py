import tensorflow as tf


class CenterLossLayer(tf.keras.layers.Layer):
    """A layer to be added in the model if you want to use CenterLoss

    Attributes
    ----------
    centers
        The variable that keeps track of centers.

    n_classes : int
        Number of classes of the task.

    n_features : int
        The size of prelogits.
    """

    def __init__(self, n_classes, n_features, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.centers = tf.Variable(
            tf.zeros([n_classes, n_features]),
            name="centers",
            trainable=False,
            # in a distributed strategy, we want updates to this variable to be summed.
            aggregation=tf.VariableAggregation.SUM,
        )

    def call(self, x):
        # pass through layer
        return tf.identity(x)

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes, "n_features": self.n_features})
        return config


class CenterLoss(tf.keras.losses.Loss):
    """Center loss.
    Introduced in: A Discriminative Feature Learning Approach for Deep Face Recognition
    https://ydwen.github.io/papers/WenECCV16.pdf

    .. warning::

        This loss MUST NOT BE CALLED during evaluation as it will update the centers!
        This loss only works with sparse labels.
        This loss must be used with CenterLossLayer embedded into the model

    Attributes
    ----------
    alpha: float
        The moving average coefficient for updating centers in each batch.

    centers
        The variable that keeps track of centers.

    centers_layer
        The layer that keeps track of centers.

    update_centers: bool
        Update the centers? Used at training
    """

    def __init__(
        self,
        centers_layer,
        alpha=0.9,
        update_centers=True,
        name="center_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.centers_layer = centers_layer
        self.centers = self.centers_layer.centers
        self.alpha = alpha
        self.update_centers = update_centers

    def call(self, sparse_labels, prelogits):
        sparse_labels = tf.reshape(sparse_labels, (-1,))
        centers_batch = tf.gather(self.centers, sparse_labels)
        # the reduction of batch dimension will be done by the parent class
        center_loss = tf.keras.losses.mean_squared_error(prelogits, centers_batch)

        # update centers
        if self.update_centers:
            diff = (1 - self.alpha) * (centers_batch - prelogits)
            updates = tf.scatter_nd(sparse_labels[:, None], diff, self.centers.shape)
            # using assign_sub will make sure updates are added during distributed
            # training
            self.centers.assign_sub(updates)

        return center_loss
