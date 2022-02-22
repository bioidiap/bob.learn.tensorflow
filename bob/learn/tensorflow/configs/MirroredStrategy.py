import tensorflow as tf


def strategy_fn():
    print("Creating MirroredStrategy strategy.")
    strategy = tf.distribute.MirroredStrategy()
    print("MirroredStrategy strategy created.")
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    return strategy
