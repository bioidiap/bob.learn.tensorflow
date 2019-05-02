"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.multiscalecnn_arg_scope()):
    outputs, end_points = alexnet.multiscalecnn(inputs)

"""


def multiscalecnn(
    inputs,
    mode,
    num_classes=2,
    dropout_keep_prob=0.5,
    spatial_squeeze=True,
    scope="multiscalecnn",
    reuse=False,
):
    """AlexNet version 2.

    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg

    Note: All the fully_connected layers have been transformed to conv2d slim.
          To use in classification mode, resize input to 224x224. To use in fully
          convolutional mode, set spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, "multiscalecnn", [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + "_end_points"
        # Collect outputs for conv2d, fully_connected, max_pool2d, and batch_norm.
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm],
            outputs_collections=[end_points_collection],
        ):
            net = slim.conv2d(inputs, 3, 1, scope="conv1")
            net = slim.conv2d(net, 32, 3, scope="conv2a")
            net = slim.conv2d(net, 32, 3, scope="conv2b")
            net = slim.max_pool2d(net, 3, scope="pool1")
            first_scale = slim.dropout(net, dropout_keep_prob, scope="dropout1")

            net = slim.conv2d(first_scale, 64, 3, scope="conv3a")
            net = slim.conv2d(net, 64, 3, scope="conv3b")
            net = slim.max_pool2d(net, 3, scope="pool2")
            second_scale = slim.dropout(net, dropout_keep_prob, scope="dropout2")

            net = slim.conv2d(second_scale, 64, 3, scope="conv4a")
            net = slim.conv2d(net, 64, 3, scope="conv4b")
            net = slim.max_pool2d(net, 3, scope="pool3")
            third_scale = slim.dropout(net, dropout_keep_prob, scope="dropout3")

            # add 1x1 convs
            first_scale = slim.conv2d(
                first_scale, 1, 1, scope="conv5", activation_fn=None
            )
            second_scale = slim.conv2d(
                second_scale, 1, 1, scope="conv6", activation_fn=None
            )
            third_scale = slim.conv2d(
                third_scale, 1, 1, scope="conv7", activation_fn=None
            )

            # AlexNet
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding="VALID", scope="conv1")
            net = slim.conv2d(net, 192, [5, 5], scope="conv2")
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool2")
            net = slim.conv2d(net, 384, [3, 3], scope="conv3")
            net = slim.conv2d(net, 384, [3, 3], scope="conv4")
            net = slim.conv2d(net, 256, [3, 3], scope="conv5")
            net = slim.max_pool2d(net, [3, 3], 2, scope="pool5")

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=trunc_normal(0.005),
                biases_initializer=init_ops.constant_initializer(0.1),
            ):
                net = slim.conv2d(net, 4096, [5, 5], padding="VALID", scope="fc6")
                net = slim.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope="dropout6"
                )
                net = slim.conv2d(net, 4096, [1, 1], scope="fc7")
                net = slim.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope="dropout7"
                )
                net = slim.conv2d(
                    net,
                    num_classes,
                    [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    scope="fc8",
                )

            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = array_ops.squeeze(net, [1, 2], name="fc8/squeezed")
                end_points[sc.name + "/fc8"] = net
            return net, end_points


multiscalecnn.default_image_size = 224


def multiscalecnn_architecture(inputs, mode, reuse=False):
    with slim.arg_scope(
        multiscalecnn_arg_scope(is_training=mode == tf.estimator.ModeKeys.TRAIN)
    ):
        outputs, end_points = multiscalecnn(inputs)
