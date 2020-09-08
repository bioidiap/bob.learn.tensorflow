"""Utilities for TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import json
import logging
import os

import tensorflow as tf

from . import DEFAULT_FEATURE


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
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
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
        "output_types": repr(tf.compat.v1.data.get_output_types(dataset)),
        "output_shapes": repr(tf.compat.v1.data.get_output_shapes(dataset)),
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
        args = tf.nest.flatten(args)
        args = [tf.io.serialize_tensor(f) for f in args]
        tf_string = tf.py_function(serialize_example_pyfunction, args, tf.string)
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
    from tensorflow import TensorShape

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
    output_types = tf.nest.flatten(meta["output_types"])
    output_shapes = tf.nest.flatten(meta["output_shapes"])
    feature_description = {}
    for i in range(len(output_types)):
        key = f"feature{i}"
        feature_description[key] = tf.io.FixedLenFeature([], tf.string)

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        args = tf.io.parse_single_example(
            serialized=example_proto, features=feature_description
        )
        args = tf.nest.flatten(args)
        args = [tf.io.parse_tensor(v, t) for v, t in zip(args, output_types)]
        args = [tf.reshape(v, s) for v, s in zip(args, output_shapes)]
        return tf.nest.pack_sequence_as(meta["output_types"], args)

    return raw_dataset.map(_parse_function)


# def write_a_sample(writer, data, label, key, feature=None, size_estimate=False):
#     if feature is None:
#         feature = {
#             "data": bytes_feature(data.tostring()),
#             "label": int64_feature(label),
#             "key": bytes_feature(key),
#         }

#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     example = example.SerializeToString()
#     if not size_estimate:
#         writer.write(example)
#     return sys.getsizeof(example)


# def example_parser(serialized_example, feature, data_shape, data_type):
#     """
#     Parses a single tf.Example into image and label tensors.

#     """
#     # Decode the record read by the reader
#     features = tf.io.parse_single_example(
#         serialized=serialized_example, features=feature
#     )
#     # Convert the image data from string back to the numbers
#     image = tf.io.decode_raw(features["data"], data_type)
#     # Cast label data into int64
#     label = tf.cast(features["label"], tf.int64)
#     # Reshape image data into the original shape
#     image = tf.reshape(image, data_shape)
#     key = tf.cast(features["key"], tf.string)
#     return image, label, key


# def image_augmentation_parser(
#     serialized_example,
#     feature,
#     data_shape,
#     data_type,
#     gray_scale=False,
#     output_shape=None,
#     random_flip=False,
#     random_brightness=False,
#     random_contrast=False,
#     random_saturation=False,
#     random_rotate=False,
#     per_image_normalization=True,
#     random_gamma=False,
#     random_crop=False,
# ):
#     """
#     Parses a single tf.Example into image and label tensors.

#     """
#     # Decode the record read by the reader
#     features = tf.io.parse_single_example(
#         serialized=serialized_example, features=feature
#     )
#     # Convert the image data from string back to the numbers
#     image = tf.io.decode_raw(features["data"], data_type)

#     # Reshape image data into the original shape
#     image = tf.reshape(image, data_shape)

#     # Applying image augmentation
#     image = append_image_augmentation(
#         image,
#         gray_scale=gray_scale,
#         output_shape=output_shape,
#         random_flip=random_flip,
#         random_brightness=random_brightness,
#         random_contrast=random_contrast,
#         random_saturation=random_saturation,
#         random_rotate=random_rotate,
#         per_image_normalization=per_image_normalization,
#         random_gamma=random_gamma,
#         random_crop=random_crop,
#     )

#     # Cast label data into int64
#     label = tf.cast(features["label"], tf.int64)
#     key = tf.cast(features["key"], tf.string)

#     return image, label, key


