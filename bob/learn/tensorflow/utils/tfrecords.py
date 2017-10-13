from functools import partial
import tensorflow as tf


DEFAULT_FEATURE = {'train/data': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}


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
        feature = DEFAULT_FEATURE
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return example_parser(serialized_example, feature, data_shape, data_type)


def create_dataset_from_records(tfrecord_filenames, data_shape, data_type,
                                feature=None):
    if feature is None:
        feature = DEFAULT_FEATURE
    dataset = tf.contrib.data.TFRecordDataset(tfrecord_filenames)
    parser = partial(example_parser, feature=feature, data_shape=data_shape,
                     data_type=data_type)
    dataset = dataset.map(parser)
    return dataset


def shuffle_data_and_labels(tfrecord_filenames, data_shape, data_type,
                            batch_size, epochs=None, buffer_size=10**3):
    dataset = create_dataset_from_records(tfrecord_filenames, data_shape,
                                          data_type)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat(epochs)

    datas, labels = dataset.make_one_shot_iterator().get_next()
    return datas, labels


def batch_data_and_labels(tfrecord_filenames, data_shape, data_type,
                          batch_size, epochs=1):
    dataset = create_dataset_from_records(tfrecord_filenames, data_shape,
                                          data_type)
    dataset = dataset.batch(batch_size).repeat(epochs)

    datas, labels = dataset.make_one_shot_iterator().get_next()
    return datas, labels
