#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import pkg_resources
import numpy
import tensorflow as tf
from bob.learn.tensorflow.dataset.generator import dataset_using_generator

data_shape = (250, 250, 3)
output_shape = (50, 50)
data_type = tf.float32
batch_size = 2
validation_batch_size = 250
epochs = 1

# Trainer logits
filenames = [
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m301_01_p01_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m301_01_p02_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m301_01_p01_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m301_01_p02_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m301_01_p01_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m301_01_p02_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m304_01_p01_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m304_02_f12_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m304_01_p01_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m304_02_f12_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m304_01_p01_i0_0.png"
    ),
    pkg_resources.resource_filename(
        __name__, "data/dummy_image_database/m304_02_f12_i0_0.png"
    ),
]
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_dataset_using_generator():
    def reader(f):
        key = 0
        label = 0
        yield {"data": f, "key": key}, label

    shape = (2, 2, 1)
    samples = [numpy.ones(shape, dtype="float32") * i for i in range(10)]

    with tf.compat.v1.Session() as session:
        dataset = dataset_using_generator(samples, reader, multiple_samples=True)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        for i in range(11):
            try:
                sample = session.run(iterator)
                assert sample[0]["data"].shape == shape
                assert numpy.allclose(sample[0]["data"], samples[i])
            except tf.errors.OutOfRangeError:
                break
