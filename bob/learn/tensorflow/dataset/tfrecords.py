"""Utilities for TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import json
import logging
import os
import sys

import tensorflow as tf

from . import append_image_augmentation, DEFAULT_FEATURE


logger = logging.getLogger(__name__)
TFRECORDS_EXT = ".tfrecords"


def tfrecord_name_and_json_name(output):
    output = normalize_tfrecords_path(output)
    json_output = output[: -len(TFRECORDS_EXT)] + ".json"
    return output, json_output


def normalize_tfrecords_path(output):
    if not output.endswith(TFRECORDS_EXT):
        output += TFRECORDS_EXT
    return output


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def dataset_to_tfrecord(dataset, output):
    """Writes a tf.data.Dataset into a TFRecord file.

    Parameters
    ----------
    dataset : ``tf.data.Dataset``
        The tf.data.Dataset that you want to write into a TFRecord file.
    output : str
        Path to the TFRecord file. Besides this file, a .json file is also created.
        This json file is needed when you want to convert the TFRecord file back into
        a dataset.

    Returns
    -------
    ``tf.Operation``
        A tf.Operation that, when run, writes contents of dataset to a file. When
        running in eager mode, calling this function will write the file. Otherwise, you
        have to call session.run() on the returned operation.
    """
    output, json_output = tfrecord_name_and_json_name(output)
    # dump the structure so that we can read it back
    meta = {
        "output_types": repr(dataset.output_types),
        "output_shapes": repr(dataset.output_shapes),
    }
    with open(json_output, "w") as f:
        json.dump(meta, f)

    # create a custom map function that serializes the dataset
    def serialize_example_pyfunction(*args):
        feature = {}
        for i, f in enumerate(args):
            key = f"feature{i}"
            feature[key] = bytes_feature(f)
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def tf_serialize_example(*args):
        args = tf.contrib.framework.nest.flatten(args)
        args = [tf.serialize_tensor(f) for f in args]
        tf_string = tf.py_func(serialize_example_pyfunction, args, tf.string)
        return tf.reshape(tf_string, ())  # The result is a scalar

    dataset = dataset.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(output)
    return writer.write(dataset)


def dataset_from_tfrecord(tfrecord, num_parallel_reads=None):
    """Reads TFRecords and returns a dataset.
    The TFRecord file must have been created using the :any:`dataset_to_tfrecord`
    function.

    Parameters
    ----------
    tfrecord : str or list
        Path to the TFRecord file. Pass a list if you are sure several tfrecords need
        the same map function.
    num_parallel_reads: int
        A `tf.int64` scalar representing the number of files to read in parallel.
        Defaults to reading files sequentially.

    Returns
    -------
    ``tf.data.Dataset``
        A dataset that contains the data from the TFRecord file.
    """
    # these imports are needed so that eval can work
    from tensorflow import TensorShape, Dimension

    if isinstance(tfrecord, str):
        tfrecord = [tfrecord]
    tfrecord = [tfrecord_name_and_json_name(path) for path in tfrecord]
    json_output = tfrecord[0][1]
    tfrecord = [path[0] for path in tfrecord]
    raw_dataset = tf.data.TFRecordDataset(
        tfrecord, num_parallel_reads=num_parallel_reads
    )

    with open(json_output) as f:
        meta = json.load(f)
    for k, v in meta.items():
        meta[k] = eval(v)
    output_types = tf.contrib.framework.nest.flatten(meta["output_types"])
    output_shapes = tf.contrib.framework.nest.flatten(meta["output_shapes"])
    feature_description = {}
    for i in range(len(output_types)):
        key = f"feature{i}"
        feature_description[key] = tf.FixedLenFeature([], tf.string)

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        args = tf.parse_single_example(example_proto, feature_description)
        args = tf.contrib.framework.nest.flatten(args)
        args = [tf.parse_tensor(v, t) for v, t in zip(args, output_types)]
        args = [tf.reshape(v, s) for v, s in zip(args, output_shapes)]
        return tf.contrib.framework.nest.pack_sequence_as(meta["output_types"], args)

    return raw_dataset.map(_parse_function)


def write_a_sample(writer, data, label, key, feature=None, size_estimate=False):
    if feature is None:
        feature = {
            "data": bytes_feature(data.tostring()),
            "label": int64_feature(label),
            "key": bytes_feature(key),
        }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example.SerializeToString()
    if not size_estimate:
        writer.write(example)
    return sys.getsizeof(example)


def example_parser(serialized_example, feature, data_shape, data_type):
    """
  Parses a single tf.Example into image and label tensors.

  """
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features["data"], data_type)
    # Cast label data into int64
    label = tf.cast(features["label"], tf.int64)
    # Reshape image data into the original shape
    image = tf.reshape(image, data_shape)
    key = tf.cast(features["key"], tf.string)
    return image, label, key


def image_augmentation_parser(
    serialized_example,
    feature,
    data_shape,
    data_type,
    gray_scale=False,
    output_shape=None,
    random_flip=False,
    random_brightness=False,
    random_contrast=False,
    random_saturation=False,
    random_rotate=False,
    per_image_normalization=True,
    random_gamma=False,
    random_crop=False,
):
    """
  Parses a single tf.Example into image and label tensors.

  """
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features["data"], data_type)

    # Reshape image data into the original shape
    image = tf.reshape(image, data_shape)

    # Applying image augmentation
    image = append_image_augmentation(
        image,
        gray_scale=gray_scale,
        output_shape=output_shape,
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast,
        random_saturation=random_saturation,
        random_rotate=random_rotate,
        per_image_normalization=per_image_normalization,
        random_gamma=random_gamma,
        random_crop=random_crop,
    )

    # Cast label data into int64
    label = tf.cast(features["label"], tf.int64)
    key = tf.cast(features["key"], tf.string)

    return image, label, key


def read_and_decode(filename_queue, data_shape, data_type=tf.float32, feature=None):
    """
  Simples parse possible for a tfrecord.
  It assumes that you have the pair **train/data** and **train/label**
  """

    if feature is None:
        feature = DEFAULT_FEATURE
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return example_parser(serialized_example, feature, data_shape, data_type)


def create_dataset_from_records(
    tfrecord_filenames, data_shape, data_type, feature=None
):
    """
  Create dataset from a list of tf-record files

  **Parameters**

     tfrecord_filenames:
        List containing the tf-record paths

     data_shape:
        Samples shape saved in the tf-record

     data_type:
        tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

     feature:

  """

    if feature is None:
        feature = DEFAULT_FEATURE
    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    parser = partial(
        example_parser, feature=feature, data_shape=data_shape, data_type=data_type
    )
    dataset = dataset.map(parser)
    return dataset


def create_dataset_from_records_with_augmentation(
    tfrecord_filenames,
    data_shape,
    data_type,
    feature=None,
    gray_scale=False,
    output_shape=None,
    random_flip=False,
    random_brightness=False,
    random_contrast=False,
    random_saturation=False,
    random_rotate=False,
    per_image_normalization=True,
    random_gamma=False,
    random_crop=False,
):
    """
  Create dataset from a list of tf-record files

  **Parameters**

     tfrecord_filenames:
        List containing the tf-record paths

     data_shape:
        Samples shape saved in the tf-record

     data_type:
        tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

     feature:

  """

    if feature is None:
        feature = DEFAULT_FEATURE
    if isinstance(tfrecord_filenames, str) and os.path.isdir(tfrecord_filenames):
        tfrecord_filenames = [
            os.path.join(tfrecord_filenames, f) for f in os.listdir(tfrecord_filenames)
        ]
    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    parser = partial(
        image_augmentation_parser,
        feature=feature,
        data_shape=data_shape,
        data_type=data_type,
        gray_scale=gray_scale,
        output_shape=output_shape,
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast,
        random_saturation=random_saturation,
        random_rotate=random_rotate,
        per_image_normalization=per_image_normalization,
        random_gamma=random_gamma,
        random_crop=random_crop,
    )
    dataset = dataset.map(parser)
    return dataset


def shuffle_data_and_labels_image_augmentation(
    tfrecord_filenames,
    data_shape,
    data_type,
    batch_size,
    epochs=None,
    buffer_size=10 ** 3,
    gray_scale=False,
    output_shape=None,
    random_flip=False,
    random_brightness=False,
    random_contrast=False,
    random_saturation=False,
    random_rotate=False,
    per_image_normalization=True,
    random_gamma=False,
    random_crop=False,
    drop_remainder=False,
):
    """Dump random batches from a list of tf-record files and applies some image augmentation

    Parameters
    ----------

      tfrecord_filenames:
        List containing the tf-record paths

      data_shape:
        Samples shape saved in the tf-record

      data_type:
        tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

      batch_size:
        Size of the batch

      epochs:
        Number of epochs to be batched

      buffer_size:
        Size of the shuffle bucket

      gray_scale:
        Convert to gray scale?

      output_shape:
        If set, will randomly crop the image given the output shape

      random_flip:
        Randomly flip an image horizontally  (https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right)

      random_brightness:
        Adjust the brightness of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_brightness)

      random_contrast:
        Adjust the contrast of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_contrast)

      random_saturation:
        Adjust the saturation of an RGB image by a random factor (https://www.tensorflow.org/api_docs/python/tf/image/random_saturation)

      random_rotate:
        Randomly rotate face images between -5 and 5 degrees

      per_image_normalization:
        Linearly scales image to have zero mean and unit norm.

      drop_remainder:
        If True, the last remaining batch that has smaller size than batch_size will be dropped.
    """

    dataset = create_dataset_from_records_with_augmentation(
        tfrecord_filenames,
        data_shape,
        data_type,
        gray_scale=gray_scale,
        output_shape=output_shape,
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast,
        random_saturation=random_saturation,
        random_rotate=random_rotate,
        per_image_normalization=per_image_normalization,
        random_gamma=random_gamma,
        random_crop=random_crop,
    )

    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.repeat(epochs)

    dataset = dataset.map(lambda d, l, k: ({"data": d, "key": k}, l))

    return dataset


def shuffle_data_and_labels(
    tfrecord_filenames,
    data_shape,
    data_type,
    batch_size,
    epochs=None,
    buffer_size=10 ** 3,
):
    """
  Dump random batches from a list of tf-record files

  **Parameters**

     tfrecord_filenames:
        List containing the tf-record paths

     data_shape:
        Samples shape saved in the tf-record

     data_type:
        tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

     batch_size:
        Size of the batch

     epochs:
         Number of epochs to be batched

     buffer_size:
          Size of the shuffle bucket

  """

    dataset = create_dataset_from_records(tfrecord_filenames, data_shape, data_type)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat(epochs)

    data, labels, key = dataset.make_one_shot_iterator().get_next()
    features = dict()
    features["data"] = data
    features["key"] = key

    return features, labels


def batch_data_and_labels(
    tfrecord_filenames, data_shape, data_type, batch_size, epochs=1
):
    """
  Dump in order batches from a list of tf-record files

  Parameters
  ----------

     tfrecord_filenames:
        List containing the tf-record paths

     data_shape:
        Samples shape saved in the tf-record

     data_type:
        tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

     batch_size:
        Size of the batch

     epochs:
         Number of epochs to be batched

  """
    dataset = create_dataset_from_records(tfrecord_filenames, data_shape, data_type)
    dataset = dataset.batch(batch_size).repeat(epochs)

    data, labels, key = dataset.make_one_shot_iterator().get_next()
    features = dict()
    features["data"] = data
    features["key"] = key

    return features, labels


def batch_data_and_labels_image_augmentation(
    tfrecord_filenames,
    data_shape,
    data_type,
    batch_size,
    epochs=1,
    gray_scale=False,
    output_shape=None,
    random_flip=False,
    random_brightness=False,
    random_contrast=False,
    random_saturation=False,
    random_rotate=False,
    per_image_normalization=True,
    random_gamma=False,
    random_crop=False,
    drop_remainder=False,
):
    """
    Dump in order batches from a list of tf-record files

    Parameters
    ----------

       tfrecord_filenames:
          List containing the tf-record paths

       data_shape:
          Samples shape saved in the tf-record

       data_type:
          tf data type(https://www.tensorflow.org/versions/r0.12/resources/dims_types#data_types)

       batch_size:
          Size of the batch

       epochs:
           Number of epochs to be batched

       drop_remainder:
           If True, the last remaining batch that has smaller size than batch_size will be dropped.
    """

    dataset = create_dataset_from_records_with_augmentation(
        tfrecord_filenames,
        data_shape,
        data_type,
        gray_scale=gray_scale,
        output_shape=output_shape,
        random_flip=random_flip,
        random_brightness=random_brightness,
        random_contrast=random_contrast,
        random_saturation=random_saturation,
        random_rotate=random_rotate,
        per_image_normalization=per_image_normalization,
        random_gamma=random_gamma,
        random_crop=random_crop,
    )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.repeat(epochs)

    data, labels, key = dataset.make_one_shot_iterator().get_next()
    features = dict()
    features["data"] = data
    features["key"] = key

    return features, labels


def describe_tf_record(tf_record_path, shape, batch_size=1):
    """
  Describe the number of samples and the number of classes of a tf-record

  Parameters
  ----------

  tf_record_path: str
    Base path containing your tf-record files

  shape: tuple
     Shape inside of the tf-record

  batch_size: int
    Well, batch size


  Returns
  -------

  n_samples: int
     Total number of samples

  n_classes: int
     Total number of classes

  """

    tf_records = [os.path.join(tf_record_path, f) for f in os.listdir(tf_record_path)]
    filename_queue = tf.train.string_input_producer(
        tf_records, num_epochs=1, name="input"
    )

    feature = {
        "data": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
        "key": tf.FixedLenFeature([], tf.string),
    }

    # Define a reader and read the next record
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features["data"], tf.uint8)

    # Cast label data into int32
    label = tf.cast(features["label"], tf.int64)
    img_name = tf.cast(features["key"], tf.string)

    # Reshape image data into the original shape
    image = tf.reshape(image, shape)

    # Getting the batches in order
    data_ph, label_ph, img_name_ph = tf.train.batch(
        [image, label, img_name],
        batch_size=batch_size,
        capacity=1000,
        num_threads=5,
        name="shuffle_batch",
    )

    # Start the reading
    session = tf.Session()
    tf.local_variables_initializer().run(session=session)
    tf.global_variables_initializer().run(session=session)

    # Preparing the batches
    thread_pool = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=thread_pool, sess=session)

    logger.info("Counting in %s", tf_record_path)
    labels = set()
    counter = 0
    try:
        while True:
            _, label, _ = session.run([data_ph, label_ph, img_name_ph])
            counter += len(label)

            for i in set(label):
                labels.add(i)

    except tf.errors.OutOfRangeError:
        pass

    thread_pool.request_stop()
    return counter, len(labels)
