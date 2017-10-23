import tensorflow as tf
from ..utils import get_available_gpus, to_channels_first


def architecture(input_layer, mode=tf.estimator.ModeKeys.TRAIN,
                 kernerl_size=(3, 3), n_classes=2):
    data_format = 'channels_last'
    if len(get_available_gpus()) != 0:
        # When running on GPU, transpose the data from channels_last (NHWC) to
        # channels_first (NCHW) to improve performance. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        input_layer = to_channels_first('input_layer')
        data_format = 'channels_first'

    # Convolutional Layer #1
    # Computes 32 features using a kernerl_size filter with ReLU activation.
    # Padding is added to preserve width and height.
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=kernerl_size,
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,
                                    data_format=data_format)

    # Convolutional Layer #2
    # Computes 64 features using a kernerl_size filter.
    # Padding is added to preserve width and height.
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=kernerl_size,
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,
                                    data_format=data_format)

    # Flatten tensor into a batch of vectors
    # TODO: use tf.layers.flatten in tensorflow 1.4 and above
    pool2_flat = tf.contrib.layers.flatten(pool2)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout, units=n_classes)

    return logits


def model_fn(features, labels, mode, params=None, config=None):
    """Model function for CNN."""
    params = params or {}
    learning_rate = params.get('learning_rate', 1e-5)
    kernerl_size = params.get('kernerl_size', (3, 3))
    n_classes = params.get('n_classes', 2)

    data = features['data']
    keys = features['keys']
    logits = architecture(
        data, mode, kernerl_size=kernerl_size, n_classes=n_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        'keys': keys,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    with tf.name_scope('train_metrics'):
        # Create a tensor named train_loss for logging purposes
        tf.summary.scalar('train_loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
