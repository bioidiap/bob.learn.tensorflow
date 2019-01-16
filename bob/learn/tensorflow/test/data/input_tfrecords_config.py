import tensorflow as tf
from bob.learn.tensorflow.dataset.tfrecords import shuffle_data_and_labels, \
    batch_data_and_labels

tfrecord_filenames = ['%(tfrecord_filenames)s']
data_shape = (1, 112, 92)  # size of atnt images
data_type = tf.uint8
batch_size = 2
epochs = 2


def train_input_fn():
    return shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                   batch_size, epochs=epochs)


def eval_input_fn():
    return batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                                 batch_size, epochs=1)


# config for train_and_evaluate
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=200)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
