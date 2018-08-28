import collections
import tensorflow as tf
from .utils import is_trainable
from ..estimators import get_trainable_variables


def create_conv_layer(inputs,
                      mode,
                      data_format,
                      endpoints,
                      number,
                      filters,
                      kernel_size,
                      pool_size,
                      pool_strides,
                      add_batch_norm=False,
                      trainable_variables=None,
                      use_bias_with_batch_norm=True):
    bn_axis = 1 if data_format.lower() == 'channels_first' else 3
    training = mode == tf.estimator.ModeKeys.TRAIN

    if add_batch_norm:
        activation = None
    else:
        activation = tf.nn.relu

    name = 'conv{}'.format(number)
    trainable = is_trainable(name, trainable_variables)
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=activation,
        data_format=data_format,
        trainable=trainable,
        use_bias=((not add_batch_norm) or use_bias_with_batch_norm),
    )
    endpoints[name] = conv

    if add_batch_norm:
        name = 'bn{}'.format(number)
        trainable = is_trainable(name, trainable_variables)
        bn = tf.layers.batch_normalization(
            conv, axis=bn_axis, training=training, trainable=trainable,
            scale=use_bias_with_batch_norm)
        endpoints[name] = bn

        name = 'activation{}'.format(number)
        bn_act = tf.nn.relu(bn)
        endpoints[name] = bn_act
    else:
        bn_act = conv

    name = 'pool{}'.format(number)
    pool = tf.layers.max_pooling2d(
        inputs=bn_act,
        pool_size=pool_size,
        strides=pool_strides,
        padding='same',
        data_format=data_format)
    endpoints[name] = pool

    return pool


def base_architecture(input_layer,
                      mode=tf.estimator.ModeKeys.TRAIN,
                      kernerl_size=(3, 3),
                      data_format='channels_last',
                      add_batch_norm=False,
                      trainable_variables=None,
                      use_bias_with_batch_norm=True,
                      **kwargs):
    training = mode == tf.estimator.ModeKeys.TRAIN
    # Keep track of all the endpoints
    endpoints = {}

    # Convolutional Layer #1
    # Computes 32 features using a kernerl_size filter with ReLU
    # activation.
    # Padding is added to preserve width and height.
    pool1 = create_conv_layer(
        inputs=input_layer,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=1,
        filters=32,
        kernel_size=kernerl_size,
        pool_size=(2, 2),
        pool_strides=2,
        add_batch_norm=add_batch_norm,
        trainable_variables=trainable_variables,
        use_bias_with_batch_norm=use_bias_with_batch_norm,
    )

    # Convolutional Layer #2
    # Computes 64 features using a kernerl_size filter.
    # Padding is added to preserve width and height.
    pool2 = create_conv_layer(
        inputs=pool1,
        mode=mode,
        data_format=data_format,
        endpoints=endpoints,
        number=2,
        filters=64,
        kernel_size=kernerl_size,
        pool_size=(2, 2),
        pool_strides=2,
        add_batch_norm=add_batch_norm,
        trainable_variables=trainable_variables,
        use_bias_with_batch_norm=use_bias_with_batch_norm,
    )

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.layers.flatten(pool2)
    endpoints['pool2_flat'] = pool2_flat

    # Dense Layer
    # Densely connected layer with 1024 neurons
    if add_batch_norm:
        activation = None
    else:
        activation = tf.nn.relu

    name = 'dense'
    trainable = is_trainable(name, trainable_variables)
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=activation,
        trainable=trainable,
        use_bias=((not add_batch_norm) or use_bias_with_batch_norm),
    )
    endpoints[name] = dense

    if add_batch_norm:
        name = 'bn{}'.format(3)
        trainable = is_trainable(name, trainable_variables)
        bn = tf.layers.batch_normalization(
            dense, axis=1, training=training, trainable=trainable,
            scale=use_bias_with_batch_norm)
        endpoints[name] = bn

        name = 'activation{}'.format(3)
        bn_act = tf.nn.relu(bn)
        endpoints[name] = bn_act
    else:
        bn_act = dense

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=bn_act, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    endpoints['dropout'] = dropout

    return dropout, endpoints


def new_architecture(
        input_layer,
        mode=tf.estimator.ModeKeys.TRAIN,
        kernerl_size=(3, 3),
        data_format='channels_last',
        add_batch_norm=True,
        trainable_variables=None,
        use_bias_with_batch_norm=False,
        reuse=False,
        **kwargs):
    with tf.variable_scope('SimpleCNN', reuse=reuse):
        return base_architecture(
            input_layer=input_layer,
            mode=mode,
            kernerl_size=kernerl_size,
            data_format=data_format,
            add_batch_norm=add_batch_norm,
            trainable_variables=trainable_variables,
            use_bias_with_batch_norm=use_bias_with_batch_norm,
            **kwargs)


def architecture(input_layer,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 kernerl_size=(3, 3),
                 n_classes=2,
                 data_format='channels_last',
                 reuse=False,
                 add_batch_norm=False,
                 trainable_variables=None,
                 **kwargs):

    with tf.variable_scope('SimpleCNN', reuse=reuse):

        dropout, endpoints = base_architecture(
            input_layer,
            mode,
            kernerl_size,
            data_format,
            add_batch_norm=add_batch_norm,
            trainable_variables=trainable_variables)
        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, n_classes]
        name = 'logits'
        trainable = is_trainable(name, trainable_variables)
        logits = tf.layers.dense(
            inputs=dropout, units=n_classes, trainable=trainable)
        endpoints[name] = logits

    return logits, endpoints


def model_fn(features, labels, mode, params=None, config=None):
    """Model function for CNN."""
    data = features['data']
    key = features['key']

    params = params or {}
    learning_rate = params.get('learning_rate', 1e-5)
    apply_moving_averages = params.get('apply_moving_averages', False)
    extra_checkpoint = params.get('extra_checkpoint')
    trainable_variables = get_trainable_variables(extra_checkpoint)
    loss_weights = params.get('loss_weights', 1.0)
    add_histograms = params.get('add_histograms')
    nnet_optimizer = params.get('nnet_optimizer') or 'sgd'

    arch_kwargs = {
        'kernerl_size': params.get('kernerl_size', None),
        'n_classes': params.get('n_classes', None),
        'data_format': params.get('data_format', None),
        'add_batch_norm': params.get('add_batch_norm', None),
        'trainable_variables': trainable_variables,
    }
    arch_kwargs = {k: v for k, v in arch_kwargs.items() if v is not None}

    logits, _ = architecture(data, mode, **arch_kwargs)

    # restore the model from an extra_checkpoint
    if extra_checkpoint is not None and mode == tf.estimator.ModeKeys.TRAIN:
        tf.train.init_from_checkpoint(
            ckpt_dir_or_file=extra_checkpoint["checkpoint_path"],
            assignment_map=extra_checkpoint["scopes"],
        )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        'key': key,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    metrics = {'accuracy': accuracy}

    global_step = tf.train.get_or_create_global_step()

    # Compute the moving average of all individual losses and the total loss.
    if apply_moving_averages and mode == tf.estimator.ModeKeys.TRAIN:
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())
    else:
        variable_averages_op = tf.no_op(name='noop')

    if mode == tf.estimator.ModeKeys.TRAIN:
        # for batch normalization to be updated as well:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    else:
        update_ops = []

    with tf.control_dependencies([variable_averages_op] + update_ops):

        # convert weights of per sample to weights per class
        if isinstance(loss_weights, collections.Iterable):
            loss_weights = tf.gather(loss_weights, labels)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels, weights=loss_weights)

        if apply_moving_averages and mode == tf.estimator.ModeKeys.TRAIN:
            # Compute the moving average of all individual losses and the total
            # loss.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(
                tf.get_collection(tf.GraphKeys.LOSSES))
        else:
            loss_averages_op = tf.no_op(name='noop')

        if mode == tf.estimator.ModeKeys.TRAIN:

            if nnet_optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = tf.group(
                optimizer.minimize(loss, global_step=global_step),
                variable_averages_op, loss_averages_op)

            # Log accuracy and loss
            with tf.name_scope('train_metrics'):
                tf.summary.scalar('accuracy', accuracy[1])
                tf.summary.scalar('loss', loss)
                if apply_moving_averages:
                    for l in tf.get_collection(tf.GraphKeys.LOSSES):
                        tf.summary.scalar(l.op.name + "_averaged",
                                          loss_averages.average(l))

            # add histograms summaries
            if add_histograms == 'all':
                for v in tf.all_variables():
                    tf.summary.histogram(v.name, v)
            elif add_histograms == 'train':
                for v in tf.trainable_variables():
                    tf.summary.histogram(v.name, v)

        else:
            train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)
