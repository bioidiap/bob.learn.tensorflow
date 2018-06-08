from bob.db.mnist import Database
import tensorflow as tf

database = Database()


def input_fn(mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        groups = 'train'
        num_epochs = None
        shuffle = True
    else:
        groups = 'test'
        num_epochs = 1
        shuffle = True
    data, labels = database.data(groups=groups)
    return tf.estimator.inputs.numpy_input_fn(
        x={
            "data": data.astype('float32'),
            'key': labels.astype('float32')
        },
        y=labels.astype('int32'),
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=shuffle)


train_input_fn = input_fn(tf.estimator.ModeKeys.TRAIN)
eval_input_fn = input_fn(tf.estimator.ModeKeys.EVAL)
