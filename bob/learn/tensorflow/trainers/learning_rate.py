import tensorflow as tf


def exponential_decay(base_learning_rate=0.05,
                      decay_steps=1000,
                      weight_decay=0.9,
                      staircase=False):

    global_step = tf.Variable(0, trainable=False)
    return tf.train.exponential_decay(base_learning_rate=base_learning_rate,
                                      global_step=global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=weight_decay,
                                      staircase=staircase
                                      )


def constant(base_learning_rate=0.05, name="constant_learning_rate"):
    return tf.Variable(base_learning_rate, name=name)
