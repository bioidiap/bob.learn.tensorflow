import tensorflow as tf
import tensorflow.contrib.slim as slim


def append_logits(
    graph,
    n_classes,
    reuse=False,
    l2_regularizer=5e-05,
    weights_std=0.1,
    trainable_variables=None,
    name="Logits",
):
    trainable = is_trainable(name, trainable_variables)
    return slim.fully_connected(
        graph,
        n_classes,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=weights_std),
        weights_regularizer=slim.l2_regularizer(l2_regularizer),
        scope=name,
        reuse=reuse,
        trainable=trainable,
    )


def is_trainable(name, trainable_variables, mode=tf.estimator.ModeKeys.TRAIN):
    """
    Check if a variable is trainable or not

    Parameters
    ----------

    name: str
       Layer name

    trainable_variables: list
       List containing the variables or scopes to be trained.
       If None, the variable/scope is trained
    """

    # if mode is not training, so we shutdown
    if mode != tf.estimator.ModeKeys.TRAIN:
        return False

    # If None, we train by default
    if trainable_variables is None:
        return True

    # Here is my choice to shutdown the whole scope
    return name in trainable_variables
