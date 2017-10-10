import tensorflow as tf


def example_parser(serialized_example, feature, data_shape):
    """Parses a single tf.Example into image and label tensors."""
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/data'], tf.float32)
    # Cast label data into int64
    label = tf.cast(features['train/label'], tf.int64)
    # Reshape image data into the original shape
    image = tf.reshape(image, data_shape)
    return image, label


def read_and_decode(filename_queue, data_shape, feature=None):

    if feature is None:
        feature = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return example_parser(serialized_example, feature, data_shape)
