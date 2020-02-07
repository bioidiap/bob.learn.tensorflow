import tensorflow as tf


class LRN(tf.keras.layers.Lambda):
    """local response normalization with default parameters for GoogLeNet
    """

    def __init__(self, alpha=0.0001, beta=0.75, depth_radius=5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.depth_radius = depth_radius

        def lrn(inputs):
            return tf.nn.local_response_normalization(
                inputs, alpha=self.alpha, beta=self.beta, depth_radius=self.depth_radius
            )

        return super().__init__(lrn, **kwargs)


class InceptionModule(tf.keras.Model):
    """The inception module as it was introduced in:

        C. Szegedy et al., “Going deeper with convolutions,” in Proceedings of the IEEE
        Conference on Computer Vision and Pattern Recognition, 2015, pp. 1–9.
    """

    def __init__(
        self,
        filter_1x1,
        filter_3x3_reduce,
        filter_3x3,
        filter_5x5_reduce,
        filter_5x5,
        pool_proj,
        name="InceptionModule",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filter_1x1 = filter_1x1
        self.filter_3x3_reduce = filter_3x3_reduce
        self.filter_3x3 = filter_3x3
        self.filter_5x5_reduce = filter_5x5_reduce
        self.filter_5x5 = filter_5x5
        self.pool_proj = pool_proj

        self.branch1_conv1 = tf.keras.layers.Conv2D(
            filter_1x1, 1, padding="same", activation="relu", name="branch1_conv1"
        )

        self.branch2_conv1 = tf.keras.layers.Conv2D(
            filter_3x3_reduce,
            1,
            padding="same",
            activation="relu",
            name="branch2_conv1",
        )
        self.branch2_conv2 = tf.keras.layers.Conv2D(
            filter_3x3, 3, padding="same", activation="relu", name="branch2_conv2"
        )

        self.branch3_conv1 = tf.keras.layers.Conv2D(
            filter_5x5_reduce,
            1,
            padding="same",
            activation="relu",
            name="branch3_conv1",
        )
        self.branch3_conv2 = tf.keras.layers.Conv2D(
            filter_5x5, 5, padding="same", activation="relu", name="branch3_conv2"
        )

        self.branch4_pool1 = tf.keras.layers.MaxPool2D(
            3, 1, padding="same", name="branch4_pool1"
        )
        self.branch4_conv1 = tf.keras.layers.Conv2D(
            pool_proj, 1, padding="same", activation="relu", name="branch4_conv1"
        )

        self.concat = tf.keras.layers.Concatenate(
            axis=-1 if tf.keras.backend.image_data_format() == "channels_last" else -3,
            name="concat",
        )

    def call(self, inputs):
        b1 = self.branch1_conv1(inputs)

        b2 = self.branch2_conv1(inputs)
        b2 = self.branch2_conv2(b2)

        b3 = self.branch3_conv1(inputs)
        b3 = self.branch3_conv2(b3)

        b4 = self.branch4_pool1(inputs)
        b4 = self.branch4_conv1(b4)

        return self.concat([b1, b2, b3, b4])


def GoogLeNet(*, num_classes=1000, name="GoogLeNet", **kwargs):
    """GoogLeNet as depicted in Figure 3 of
    C. Szegedy et al., “Going deeper with convolutions,” in Proceedings of the IEEE
    Conference on Computer Vision and Pattern Recognition, 2015, pp. 1–9.
    and implemented in caffe:
    https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(
                64, 7, strides=2, padding="same", activation="relu", name="conv1/7x7_s2"
            ),
            tf.keras.layers.MaxPool2D(3, 2, padding="same", name="pool1/3x3_s2"),
            LRN(name="pool1/norm1"),
            tf.keras.layers.Conv2D(64, 1, padding="same", activation="relu", name="conv2/3x3_reduce"),
            tf.keras.layers.Conv2D(
                192, 3, padding="same", activation="relu", name="conv2/3x3"
            ),
            LRN(name="conv2/norm2"),
            tf.keras.layers.MaxPool2D(3, 2, padding="same", name="pool2/3x3_s2"),
            InceptionModule(64, 96, 128, 16, 32, 32, name="inception_3a"),
            InceptionModule(128, 128, 192, 32, 96, 64, name="inception_3b"),
            tf.keras.layers.MaxPool2D(3, 2, padding="same", name="pool3/3x3_s2"),
            InceptionModule(192, 96, 208, 16, 48, 64, name="inception_4a"),
            InceptionModule(160, 112, 224, 24, 64, 64, name="inception_4b"),
            InceptionModule(128, 128, 256, 24, 64, 64, name="inception_4c"),
            InceptionModule(112, 144, 288, 32, 64, 64, name="inception_4d"),
            InceptionModule(256, 160, 320, 32, 128, 128, name="inception_4e"),
            tf.keras.layers.MaxPool2D(3, 2, padding="same", name="pool4/3x3_s2"),
            InceptionModule(256, 160, 320, 32, 128, 128, name="inception_5a"),
            InceptionModule(384, 192, 384, 48, 128, 128, name="inception_5b"),
            tf.keras.layers.GlobalAvgPool2D(name="pool5"),
            tf.keras.layers.Dropout(rate=0.4, name="dropout"),
            tf.keras.layers.Dense(num_classes, name="output", activation="softmax"),
        ],
        name=name,
        **kwargs
    )

    return model


if __name__ == "__main__":
    import pkg_resources
    from tabulate import tabulate
    from bob.learn.tensorflow.utils import model_summary

    inputs = tf.keras.Input((28, 28, 192), name="input")
    model = InceptionModule(64, 96, 128, 16, 32, 32)
    outputs = model.call(inputs)
    model = tf.keras.Model(inputs, outputs)
    rows = model_summary(model, do_print=True)
    del rows[-2]
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))

    model = GoogLeNet()
    rows = model_summary(model, do_print=True)
    del rows[-2]
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))
