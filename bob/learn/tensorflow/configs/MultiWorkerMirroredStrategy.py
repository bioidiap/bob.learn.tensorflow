import tensorflow as tf


def strategy_fn():
    print("Creating MultiWorkerMirroredStrategy strategy.")
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
        )
    )
    print("MultiWorkerMirroredStrategy strategy created.")
    return strategy
