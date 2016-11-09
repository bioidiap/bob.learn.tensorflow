import tensorflow as tf


def exponential_decay(base_learning_rate=0.05,
                      decay_steps=1000,
                      weight_decay=0.9,
                      staircase=False):
    """
    Implements the exponential_decay update of the learning rate.

    https://en.wikipedia.org/wiki/Exponential_decay

    ** Parameters **

    base_learning_rate: A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
    decay_steps: A scalar int32 or int64 Tensor or a Python number. Must be positive. See the decay computation above.
    weight_decay: A scalar float32 or float64 Tensor or a Python number. The decay rate.
    staircase: Boolean. It True decay the learning rate at discrete intervals
    """

    global_step = tf.Variable(0, trainable=False)
    return tf.train.exponential_decay(base_learning_rate=base_learning_rate,
                                      global_step=global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=weight_decay,
                                      staircase=staircase
                                      )


def constant(base_learning_rate=0.05, name="constant_learning_rate"):
    """
    Create a constant learning rate

    ** Parameters **

    base_learning_rate: A scalar float32 or float64 Tensor or a Python number. The initial learning rate.

    """

    return tf.Variable(base_learning_rate, name=name)
