import tensorflow as tf

model_dir = "%(model_dir)s"
learning_rate = 0.00001


def architecture(images):
    images = tf.cast(images, tf.float32)
    logits = tf.reshape(images, [-1, 92 * 112])
    logits = tf.contrib.slim.fully_connected(inputs=logits, num_outputs=20)
    return logits


def model_fn(features, labels, mode, config):
    key = features['key']
    features = features['data']

    logits = architecture(features)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "key": key,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    metrics = {'accuracy': accuracy}

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_or_create_global_step())
        # Log accuracy and loss
        with tf.name_scope('train_metrics'):
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('loss', loss)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
