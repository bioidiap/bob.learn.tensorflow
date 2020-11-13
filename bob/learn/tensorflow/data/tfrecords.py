"""Utilities for TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

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
    from tensorflow import TensorShape  # noqa: F401

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
