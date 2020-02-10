"""Densely Connected Convolutional Networks.
Reference [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
"""

import tensorflow as tf
from bob.extension import rc

l2 = tf.keras.regularizers.l2


class ConvBlock(tf.keras.Model):
    """Convolutional Block consisting of (batchnorm->relu->conv).

    Arguments:
        num_filters: number of filters passed to a convolutional layer.
        data_format: "channels_first" or "channels_last"
        bottleneck: if True, then a 1x1 Conv is performed followed by 3x3 Conv.
        weight_decay: weight decay
        dropout_rate: dropout rate.
    """

    def __init__(
        self,
        num_filters,
        data_format,
        bottleneck,
        weight_decay=1e-4,
        dropout_rate=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bottleneck = bottleneck

        axis = -1 if data_format == "channels_last" else 1
        inter_filter = num_filters * 4
        self.num_filters = num_filters
        self.bottleneck = bottleneck
        self.dropout_rate = dropout_rate

        self.norm1 = tf.keras.layers.BatchNormalization(axis=axis, name="norm1")
        if self.bottleneck:
            self.relu1 = tf.keras.layers.Activation("relu", name="relu1")
            self.conv1 = tf.keras.layers.Conv2D(
                inter_filter,
                (1, 1),
                padding="valid",
                use_bias=False,
                data_format=data_format,
                kernel_initializer="he_normal",
                kernel_regularizer=l2(weight_decay),
                name="conv1",
            )
            self.norm2 = tf.keras.layers.BatchNormalization(axis=axis, name="norm2")

        self.relu2 = tf.keras.layers.Activation("relu", name="relu2")
        self.conv2_pad = tf.keras.layers.ZeroPadding2D(
            padding=1, data_format=data_format, name="conv2_pad"
        )
        # don't forget to set use_bias=False when using batchnorm
        self.conv2 = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            padding="valid",
            use_bias=False,
            data_format=data_format,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(weight_decay),
            name="conv2",
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name="dropout")

    def call(self, x, training=None):
        output = self.norm1(x, training=training)

        if self.bottleneck:
            output = self.relu1(output)
            output = self.conv1(output)
            output = self.norm2(output, training=training)

        output = self.relu2(output)
        output = self.conv2_pad(output)
        output = self.conv2(output)
        output = self.dropout(output, training=training)

        return output


class DenseBlock(tf.keras.Model):
    """Dense Block consisting of ConvBlocks where each block's
    output is concatenated with its input.

    Arguments:
        num_layers: Number of layers in each block.
        growth_rate: number of filters to add per conv block.
        data_format: "channels_first" or "channels_last"
        bottleneck: boolean, that decides which part of ConvBlock to call.
        weight_decay: weight decay
        dropout_rate: dropout rate.
    """

    def __init__(
        self,
        num_layers,
        growth_rate,
        data_format,
        bottleneck,
        weight_decay=1e-4,
        dropout_rate=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.bottleneck = bottleneck
        self.dropout_rate = dropout_rate
        self.axis = -1 if data_format == "channels_last" else 1

        self.blocks = []
        for i in range(int(self.num_layers)):
            self.blocks.append(
                ConvBlock(
                    growth_rate,
                    data_format,
                    bottleneck,
                    weight_decay,
                    dropout_rate,
                    name=f"conv_block_{i+1}",
                )
            )

    def call(self, x, training=None):
        for i in range(int(self.num_layers)):
            output = self.blocks[i](x, training=training)
            x = tf.keras.layers.Concatenate(axis=self.axis, name=f"concat_{i+1}")(
                [x, output]
            )

        return x


class TransitionBlock(tf.keras.Model):
    """Transition Block to reduce the number of features.

    Arguments:
        num_filters: number of filters passed to a convolutional layer.
        data_format: "channels_first" or "channels_last"
        weight_decay: weight decay
    """

    def __init__(self, num_filters, data_format, weight_decay=1e-4, **kwargs):
        super().__init__(**kwargs)
        axis = -1 if data_format == "channels_last" else 1
        self.num_filters = num_filters

        self.norm = tf.keras.layers.BatchNormalization(axis=axis, name="norm")
        self.relu = tf.keras.layers.Activation("relu", name="relu")
        self.conv = tf.keras.layers.Conv2D(
            num_filters,
            (1, 1),
            padding="valid",
            use_bias=False,
            data_format=data_format,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(weight_decay),
            name="conv",
        )
        self.pool = tf.keras.layers.AveragePooling2D(
            data_format=data_format, name="pool"
        )

    def call(self, x, training=None):
        output = self.norm(x, training=training)
        output = self.relu(output)
        output = self.conv(output)
        output = self.pool(output)
        return output


class DenseNet(tf.keras.Model):
    """Creating the Densenet Architecture.

    Arguments:
        depth_of_model: number of layers in the model.
        growth_rate: number of filters to add per conv block.
        num_of_blocks: number of dense blocks.
        output_classes: number of output classes.
        num_layers_in_each_block: number of layers in each block.
                                  If -1, then we calculate this by (depth-3)/4.
                                  If positive integer, then the it is used as the
                                    number of layers per block.
                                  If list or tuple, then this list is used directly.
        data_format: "channels_first" or "channels_last"
        bottleneck: boolean, to decide which part of conv block to call.
        compression: reducing the number of inputs(filters) to the transition block.
        weight_decay: weight decay
        rate: dropout rate.
        pool_initial: If True add a 7x7 conv with stride 2 followed by 3x3 maxpool
                      else, do a 3x3 conv with stride 1.
        include_top: If true, GlobalAveragePooling Layer and Dense layer are
                     included.
    """

    def __init__(
        self,
        depth_of_model,
        growth_rate,
        num_of_blocks,
        output_classes,
        num_layers_in_each_block,
        data_format,
        bottleneck=True,
        compression=0.5,
        weight_decay=1e-4,
        dropout_rate=0,
        pool_initial=False,
        include_top=True,
        name="DenseNet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.depth_of_model = depth_of_model
        self.growth_rate = growth_rate
        self.num_of_blocks = num_of_blocks
        self.output_classes = output_classes
        self.num_layers_in_each_block = num_layers_in_each_block
        self.data_format = data_format
        self.bottleneck = bottleneck
        self.compression = compression
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.pool_initial = pool_initial
        self.include_top = include_top

        # deciding on number of layers in each block
        if isinstance(self.num_layers_in_each_block, list) or isinstance(
            self.num_layers_in_each_block, tuple
        ):
            self.num_layers_in_each_block = list(self.num_layers_in_each_block)
        else:
            if self.num_layers_in_each_block == -1:
                if self.num_of_blocks != 3:
                    raise ValueError(
                        "Number of blocks must be 3 if num_layers_in_each_block is -1"
                    )
                if (self.depth_of_model - 4) % 3 == 0:
                    num_layers = (self.depth_of_model - 4) / 3
                    if self.bottleneck:
                        num_layers //= 2
                    self.num_layers_in_each_block = [num_layers] * self.num_of_blocks
                else:
                    raise ValueError("Depth must be 3N+4 if num_layer_in_each_block=-1")
            else:
                self.num_layers_in_each_block = [
                    self.num_layers_in_each_block
                ] * self.num_of_blocks

        axis = -1 if self.data_format == "channels_last" else 1

        # setting the filters and stride of the initial covn layer.
        if self.pool_initial:
            init_filters = (7, 7)
            stride = (2, 2)
        else:
            init_filters = (3, 3)
            stride = (1, 1)

        self.num_filters = 2 * self.growth_rate

        # first conv and pool layer
        self.conv0_pad = tf.keras.layers.ZeroPadding2D(
            padding=3, data_format=data_format, name="conv0_pad"
        )
        self.conv0 = tf.keras.layers.Conv2D(
            self.num_filters,
            init_filters,
            strides=stride,
            padding="valid",
            use_bias=False,
            data_format=self.data_format,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(self.weight_decay),
            name="conv0",
        )
        if self.pool_initial:
            self.norm0 = tf.keras.layers.BatchNormalization(axis=axis, name="norm0")
            self.relu0 = tf.keras.layers.Activation("relu", name="relu0")
            self.pool0_pad = tf.keras.layers.ZeroPadding2D(
                padding=1, data_format=data_format, name="pool0_pad"
            )
            self.pool0 = tf.keras.layers.MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding="valid",
                data_format=self.data_format,
                name="pool0",
            )

        # calculating the number of filters after each block
        num_filters_after_each_block = [self.num_filters]
        for i in range(1, self.num_of_blocks):
            temp_num_filters = num_filters_after_each_block[i - 1] + (
                self.growth_rate * self.num_layers_in_each_block[i - 1]
            )
            # using compression to reduce the number of inputs to the
            # transition block
            temp_num_filters = int(temp_num_filters * compression)
            num_filters_after_each_block.append(temp_num_filters)

        # dense block initialization
        self.dense_blocks = []
        self.transition_blocks = []
        for i in range(self.num_of_blocks):
            self.dense_blocks.append(
                DenseBlock(
                    self.num_layers_in_each_block[i],
                    growth_rate=self.growth_rate,
                    data_format=self.data_format,
                    bottleneck=self.bottleneck,
                    weight_decay=self.weight_decay,
                    dropout_rate=self.dropout_rate,
                    name=f"dense_block_{i+1}",
                )
            )
            if i + 1 < self.num_of_blocks:
                self.transition_blocks.append(
                    TransitionBlock(
                        num_filters_after_each_block[i + 1],
                        data_format=self.data_format,
                        weight_decay=self.weight_decay,
                        name=f"transition_block_{i+1}",
                    )
                )

        # Final batch norm
        self.norm5 = tf.keras.layers.BatchNormalization(axis=axis, name="norm5")
        self.relu5 = tf.keras.layers.Activation("relu", name="relu5")

        # last pooling and fc layer
        if self.include_top:
            self.last_pool = tf.keras.layers.GlobalAveragePooling2D(
                data_format=self.data_format, name="last_pool"
            )
            self.classifier = tf.keras.layers.Dense(
                self.output_classes, name="classifier"
            )

    def call(self, x, training=None):
        output = self.conv0_pad(x)
        output = self.conv0(output)

        if self.pool_initial:
            output = self.norm0(output, training=training)
            output = self.relu0(output)
            output = self.pool0_pad(output)
            output = self.pool0(output)

        for i in range(self.num_of_blocks - 1):
            output = self.dense_blocks[i](output, training=training)
            output = self.transition_blocks[i](output, training=training)

        output = self.dense_blocks[self.num_of_blocks - 1](output, training=training)
        output = self.norm5(output, training=training)
        output = self.relu5(output)

        if self.include_top:
            output = self.last_pool(output)
            output = self.classifier(output)

        return output


def densenet161(
    weights="imagenet",
    output_classes=1000,
    data_format="channels_last",
    weight_decay=1e-4,
    depth_of_model=161,
    growth_rate=48,
    num_of_blocks=4,
    num_layers_in_each_block=(6, 12, 36, 24),
    pool_initial=True,
    **kwargs,
):
    model = DenseNet(
        depth_of_model=depth_of_model,
        growth_rate=growth_rate,
        num_of_blocks=num_of_blocks,
        num_layers_in_each_block=num_layers_in_each_block,
        pool_initial=pool_initial,
        output_classes=output_classes,
        data_format=data_format,
        weight_decay=weight_decay,
        **kwargs,
    )
    if weights == "imagenet":
        model.load_weights(rc["bob.learn.tensorflow.densenet161"])
    return model


class DeepPixBiS(tf.keras.Model):
    """DeepPixBiS"""

    def __init__(self, weight_decay=1e-5, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)

        model = densenet161(
            weights=None,
            include_top=False,
            weight_decay=weight_decay,
            data_format=data_format,
        )

        # create a new model with needed layers
        self.sequential_layers = [
            model.conv0_pad,
            model.conv0,
            model.norm0,
            model.relu0,
            model.pool0_pad,
            model.pool0,
            model.dense_blocks[0],
            model.transition_blocks[0],
            model.dense_blocks[1],
            model.transition_blocks[1],
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                kernel_initializer="he_normal",
                kernel_regularizer=l2(weight_decay),
                data_format=data_format,
                name="dec",
            ),
            tf.keras.layers.Flatten(
                data_format=data_format, name="Pixel_Logits_Flatten"
            ),
            tf.keras.layers.Activation("sigmoid", name="activation"),
        ]

    def call(self, x, training=None):
        for l in self.sequential_layers:
            try:
                x = l(x, training=training)
            except TypeError:
                x = l(x)
        return x


if __name__ == "__main__":
    import pkg_resources
    from tabulate import tabulate
    from bob.learn.tensorflow.utils import model_summary

    def print_model(inputs, outputs):
        model = tf.keras.Model(inputs, outputs)
        rows = model_summary(model, do_print=True)
        del rows[-2]
        print(tabulate(rows, headers="firstrow", tablefmt="latex"))

    # inputs = tf.keras.Input((224, 224, 3), name="input")
    # model = densenet161(weights=None)
    # outputs = model.call(inputs)
    # print_model(inputs, outputs)

    # inputs = tf.keras.Input((56, 56, 96))
    # outputs = model.dense_blocks[0].call(inputs)
    # print_model(inputs, outputs)

    # inputs = tf.keras.Input((56, 56, 96))
    # outputs = model.dense_blocks[0].blocks[0].call(inputs)
    # print_model(inputs, outputs)

    # inputs = tf.keras.Input((56, 56, 384))
    # outputs = model.transition_blocks[0].call(inputs)
    # print_model(inputs, outputs)

    inputs = tf.keras.Input((224, 224, 3), name="input")
    model = DeepPixBiS()
    outputs = model.call(inputs)
    print_model(inputs, outputs)
