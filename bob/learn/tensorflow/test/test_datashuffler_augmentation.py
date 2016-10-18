#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory, Disk, SiameseDisk, TripletDisk, ImageAugmentation
import pkg_resources
from ..util import load_mnist
import os

"""
Some unit tests for the datashuffler
"""


def get_dummy_files():

    base_path = pkg_resources.resource_filename(__name__, 'data/dummy_database')
    files = []
    clients = []
    for f in os.listdir(base_path):
        if f.endswith(".hdf5"):
            files.append(os.path.join(base_path, f))
            clients.append(int(f[1:4]))

    return files, clients


def test_memory_shuffler():

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    batch_shape = [16, 28, 28, 1]

    data_augmentation = ImageAugmentation()
    data_shuffler = Memory(train_data, train_labels,
                           input_shape=batch_shape[1:],
                           scale=True,
                           batch_size=batch_shape[0],
                           data_augmentation=data_augmentation)

    batch = data_shuffler.get_batch()
    assert len(batch) == 2
    assert batch[0].shape == tuple(batch_shape)
    assert batch[1].shape[0] == batch_shape[0]


def test_siamesememory_shuffler():

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    batch_shape = [16, 28, 28, 1]
    data_augmentation = ImageAugmentation()
    data_shuffler = SiameseMemory(train_data, train_labels,
                                  input_shape=batch_shape[1:],
                                  scale=True,
                                  batch_size=batch_shape[0],
                                  data_augmentation=data_augmentation)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == tuple(batch_shape)
    assert batch[1].shape == tuple(batch_shape)
    assert batch[2].shape[0] == batch_shape[0]

    placeholders = data_shuffler.get_placeholders(name="train")
    assert placeholders[0].get_shape().as_list() == batch_shape
    assert placeholders[1].get_shape().as_list() == batch_shape
    assert placeholders[2].get_shape().as_list()[0] == batch_shape[0]


def test_tripletmemory_shuffler():

    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    batch_shape = [16, 28, 28, 1]
    data_augmentation = ImageAugmentation()
    data_shuffler = TripletMemory(train_data, train_labels,
                                  input_shape=batch_shape[1:],
                                  scale=True,
                                  batch_size=batch_shape[0],
                                  data_augmentation=data_augmentation)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == tuple(batch_shape)
    assert batch[1].shape == tuple(batch_shape)
    assert batch[2].shape == tuple(batch_shape)

    placeholders = data_shuffler.get_placeholders(name="train")
    assert placeholders[0].get_shape().as_list() == batch_shape
    assert placeholders[1].get_shape().as_list() == batch_shape
    assert placeholders[2].get_shape().as_list() == batch_shape


def test_disk_shuffler():

    train_data, train_labels = get_dummy_files()

    batch_shape = [2, 125, 125, 3]
    data_augmentation = ImageAugmentation()
    data_shuffler = Disk(train_data, train_labels,
                         input_shape=batch_shape[1:],
                         scale=True,
                         batch_size=batch_shape[0],
                         data_augmentation=data_augmentation)

    batch = data_shuffler.get_batch()

    assert len(batch) == 2
    assert batch[0].shape == tuple(batch_shape)
    assert batch[1].shape[0] == batch_shape[0]

    placeholders = data_shuffler.get_placeholders(name="train")
    assert placeholders[0].get_shape().as_list() == batch_shape
    assert placeholders[1].get_shape().as_list()[0] == batch_shape[0]


def test_siamesedisk_shuffler():

    train_data, train_labels = get_dummy_files()

    batch_shape = [2, 125, 125, 3]
    data_augmentation = ImageAugmentation()
    data_shuffler = SiameseDisk(train_data, train_labels,
                                input_shape=batch_shape[1:],
                                scale=True,
                                batch_size=batch_shape[0],
                                data_augmentation=data_augmentation)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == tuple(batch_shape)
    assert batch[1].shape == tuple(batch_shape)
    assert batch[2].shape[0] == batch_shape[0]

    placeholders = data_shuffler.get_placeholders(name="train")
    assert placeholders[0].get_shape().as_list() == batch_shape
    assert placeholders[1].get_shape().as_list() == batch_shape
    assert placeholders[2].get_shape().as_list()[0] == batch_shape[0]


def test_tripletdisk_shuffler():

    train_data, train_labels = get_dummy_files()

    batch_shape = [1, 125, 125, 3]
    data_augmentation = ImageAugmentation()
    data_shuffler = TripletDisk(train_data, train_labels,
                                input_shape=batch_shape[1:],
                                scale=True,
                                batch_size=batch_shape[0],
                                data_augmentation=data_augmentation)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == tuple(batch_shape)
    assert batch[1].shape == tuple(batch_shape)
    assert batch[2].shape == tuple(batch_shape)

    placeholders = data_shuffler.get_placeholders(name="train")
    assert placeholders[0].get_shape().as_list() == batch_shape
    assert placeholders[1].get_shape().as_list() == batch_shape
    assert placeholders[2].get_shape().as_list() == batch_shape