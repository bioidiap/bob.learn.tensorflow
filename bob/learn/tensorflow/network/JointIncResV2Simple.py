from .InceptionResnetV2 import inception_resnet_v2_batch_norm
from .InceptionResnetV1 import inception_resnet_v1_batch_norm
from .SimpleCNN import base_architecture as simplecnn_arch
import numpy as np
import tensorflow as tf


def architecture(faces, mode, face_arch='InceptionResnetV2', **kwargs):
    # construct patches inside the model
    ksizes = strides = [1, 28, 28, 1]
    rates = [1, 1, 1, 1]
    patches = tf.extract_image_patches(faces, ksizes, strides, rates, 'VALID')
    n_blocks = int(np.prod(patches.shape[1:3]))
    # n_blocks should be 25 for 160x160 faces
    patches = tf.reshape(patches, [-1, n_blocks, 28, 28, 3])

    simplecnn_kwargs = {
        'kernerl_size': (3, 3),
        'data_format': 'channels_last',
        'add_batch_norm': True,
        'use_bias_with_batch_norm': False,
    }
    simplecnn_kwargs.update(kwargs)
    endpoints = {}
    # construct simplecnn from patches
    for i in range(n_blocks):
        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('SimpleCNN', reuse=reuse):
            net, temp = simplecnn_arch(patches[:, i], mode, **simplecnn_kwargs)
        if i == 0:
            simplecnn_embeddings = net
            endpoints.update(temp)
        else:
            simplecnn_embeddings += net
    # average the embeddings of patches
    simplecnn_embeddings /= n_blocks

    # construct inception_resnet_v1 or 2 from faces
    if face_arch == 'InceptionResnetV2':
        face_embeddings, temp = inception_resnet_v2_batch_norm(
            faces, mode=mode, **kwargs)
    elif face_arch == 'InceptionResnetV1':
        face_embeddings, temp = inception_resnet_v1_batch_norm(
            faces, mode=mode, **kwargs)
    endpoints.update(temp)

    embeddings = tf.concat([simplecnn_embeddings, face_embeddings], 1)

    endpoints['final_embeddings'] = embeddings

    return embeddings, endpoints


def model_fn(features, labels, mode, params, config):
    """The model function for join face and patch PAD. The input to the model
    is 160x160 faces."""

    faces = features['data']
    key = features['key']

    # organize the parameters
    params = params or {}
    learning_rate = params.get('learning_rate', 1e-4)
    apply_moving_averages = params.get('apply_moving_averages', True)
    n_classes = params.get('n_classes', 2)
    add_histograms = params.get('add_histograms')
    face_arch = params.get('face_arch', 'InceptionResnetV2')

    embeddings, _ = architecture(faces, mode, face_arch=face_arch)

    # Logits layer
    logits = tf.layers.dense(inputs=embeddings, units=n_classes, name='logits')

    # # restore the model from an extra_checkpoint
    # if extra_checkpoint is not None and mode == tf.estimator.ModeKeys.TRAIN:
    #     tf.train.init_from_checkpoint(
    #         ckpt_dir_or_file=extra_checkpoint["checkpoint_path"],
    #         assignment_map=extra_checkpoint["scopes"],
    #     )

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

        # Calculate Loss (for both TRAIN and EVAL modes)
        cross_loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)

        loss = tf.add_n(
            [cross_loss] + regularization_losses, name="total_loss")

        if apply_moving_averages and mode == tf.estimator.ModeKeys.TRAIN:
            # Compute the moving average of all individual losses and the total
            # loss.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(
                tf.get_collection(tf.GraphKeys.LOSSES))
        else:
            loss_averages_op = tf.no_op(name='noop')

        if mode == tf.estimator.ModeKeys.TRAIN:

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
            train_op = tf.group(
                optimizer.minimize(loss, global_step=global_step),
                variable_averages_op, loss_averages_op)

            # Log accuracy and loss
            with tf.name_scope('train_metrics'):
                tf.summary.scalar('accuracy', accuracy[1])
                tf.summary.scalar('cross_entropy_loss', cross_loss)
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
