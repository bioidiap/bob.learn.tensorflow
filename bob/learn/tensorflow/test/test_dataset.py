#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import pkg_resources
import tensorflow as tf
from bob.learn.tensorflow.dataset.siamese_image import shuffle_data_and_labels_image_augmentation as siamese_batch
from bob.learn.tensorflow.dataset.triplet_image import shuffle_data_and_labels_image_augmentation as triplet_batch

data_shape = (250, 250, 3)
output_shape = (50, 50)
data_type = tf.float32
batch_size = 2
validation_batch_size = 250
epochs = 1

# Trainer logits
filenames = [
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m301_01_p02_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_01_p01_i0_0.png'),
    pkg_resources.resource_filename(
        __name__, 'data/dummy_image_database/m304_02_f12_i0_0.png')
]
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_siamese_dataset():
    data, label = siamese_batch(
        filenames,
        labels,
        data_shape,
        data_type,
        2,
        per_image_normalization=False,
        output_shape=output_shape)

    with tf.Session() as session:
        d, l = session.run([data, label])
        assert len(l) == 2
        assert d['left'].shape == (2, 50, 50, 3)
        assert d['right'].shape == (2, 50, 50, 3)


def test_triplet_dataset():
    data = triplet_batch(
        filenames,
        labels,
        data_shape,
        data_type,
        2,
        per_image_normalization=False,
        output_shape=output_shape)
    with tf.Session() as session:
        d = session.run([data])[0]
        assert len(d.keys()) == 3
        assert d['anchor'].shape == (2, 50, 50, 3)
        assert d['positive'].shape == (2, 50, 50, 3)
        assert d['negative'].shape == (2, 50, 50, 3)
