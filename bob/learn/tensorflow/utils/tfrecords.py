import tensorflow as tf


def example_parser(serialized_example, feature, data_shape, data_type):
    """Parses a single tf.Example into image and label tensors."""
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/data'], data_type)
    # Cast label data into int64
    label = tf.cast(features['train/label'], tf.int64)
    # Reshape image data into the original shape
    image = tf.reshape(image, data_shape)
    return image, label


def read_and_decode(filename_queue, data_shape, data_type=tf.float32,
                    feature=None):
    if feature is None:
        feature = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return example_parser(serialized_example, feature, data_shape, data_type)


def _read_data_and_labesl(tfrecord_filenames, data_shape, data_type,
                          epochs=None):

    filename_queue = tf.train.string_input_producer(
        tfrecord_filenames, num_epochs=epochs, name="tfrecord_filenames")

    data, label = read_and_decode(filename_queue, data_shape, data_type)
    return data, label


def shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                            batch_size, epochs=None, capacity=10**3,
                            min_after_dequeue=None, num_threads=1):
    if min_after_dequeue is None:
        min_after_dequeue = capacity // 2
    data, label = _read_data_and_labesl(
        tfrecord_filenames, data_shape, data_type, epochs)

    datas, labels = tf.train.shuffle_batch(
        [data, label], batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads, name="shuffle_batch")
    return datas, labels


def batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                          batch_size, epochs=1, capacity=10**3, num_threads=1):
    data, label = _read_data_and_labesl(
        tfrecord_filenames, data_shape, data_type, epochs)

    datas, labels = tf.train.batch(
        [data, label], batch_size=batch_size,
        capacity=capacity,
        num_threads=num_threads, name="batch")
    return datas, labels
