#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import numpy
from bob.learn.tensorflow.datashuffler import Memory, SiameseMemory, TripletMemory, Disk, SiameseDisk, TripletDisk, \
    TripletWithFastSelectionDisk, TripletWithSelectionDisk, DiskAudio
import pkg_resources
from bob.learn.tensorflow.utils import load_mnist
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


def get_dummy_audiofiles():

    base_path = pkg_resources.resource_filename(__name__, 'data/dummy_audio')
    files = []
    labels = []
    for f in os.listdir(base_path):
        if f.endswith(".wav"):
            files.append(os.path.join(base_path, f))
            labels.append((1 if 'attack' in f else 0))

    return files, labels


def test_memory_shuffler():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    shape = [None, 28, 28, 1]

    data_shuffler = Memory(train_data, train_labels,
                           input_shape=shape,
                           batch_size=16)
    batch = data_shuffler.get_batch()

    assert len(batch) == 2
    assert batch[0].shape == (16, 28, 28, 1)


def test_siamesememory_shuffler():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    batch_shape = [None, 28, 28, 1]

    data_shuffler = SiameseMemory(train_data, train_labels,
                                  input_shape=batch_shape,
                                  batch_size=16)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == (16, 28, 28, 1)
    assert batch[1].shape == (16, 28, 28, 1)


def test_tripletmemory_shuffler():
    train_data, train_labels, validation_data, validation_labels = load_mnist()
    train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    batch_shape = [None, 28, 28, 1]

    data_shuffler = TripletMemory(train_data, train_labels,
                                  input_shape=batch_shape,
                                  batch_size=16)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == (16, 28, 28, 1)
    assert batch[1].shape == (16, 28, 28, 1)
    assert batch[2].shape == (16, 28, 28, 1)


def test_disk_shuffler():
    train_data, train_labels = get_dummy_files()
    batch_shape = [None, 250, 250, 3]

    data_shuffler = Disk(train_data, train_labels,
                         input_shape=batch_shape,
                         batch_size=2)

    batch = data_shuffler.get_batch()

    assert len(batch) == 2
    assert batch[0].shape == (2, 250, 250, 3)


def test_siamesedisk_shuffler():
    train_data, train_labels = get_dummy_files()

    batch_shape = [None, 250, 250, 3]

    data_shuffler = SiameseDisk(train_data, train_labels,
                                input_shape=batch_shape,
                                batch_size=2)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == (2, 250, 250, 3)
    assert batch[1].shape == (2, 250, 250, 3)


def test_tripletdisk_shuffler():
    train_data, train_labels = get_dummy_files()

    batch_shape = [None, 250, 250, 3]

    data_shuffler = TripletDisk(train_data, train_labels,
                                input_shape=batch_shape,
                                batch_size=1)

    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert batch[0].shape == (1, 250, 250, 3)
    assert batch[1].shape == (1, 250, 250, 3)
    assert batch[2].shape == (1, 250, 250, 3)


def test_triplet_fast_selection_disk_shuffler():
    train_data, train_labels = get_dummy_files()

    batch_shape = [None, 125, 125, 3]

    data_shuffler = TripletWithFastSelectionDisk(train_data, train_labels,
                                                 input_shape=batch_shape,
                                                 total_identities=1,
                                                 batch_size=1)

    batch = data_shuffler.get_batch()
    assert len(batch) == 3
    assert len(batch[0].shape) == len(tuple(batch_shape))
    assert len(batch[1].shape) == len(tuple(batch_shape))
    assert len(batch[2].shape) == len(tuple(batch_shape))


def test_triplet_selection_disk_shuffler():
    train_data, train_labels = get_dummy_files()

    batch_shape = [None, 125, 125, 3]

    data_shuffler = TripletWithSelectionDisk(train_data, train_labels,
                                             input_shape=batch_shape,
                                             total_identities=1,
                                             batch_size=1)
    batch = data_shuffler.get_batch()

    assert len(batch) == 3
    assert len(batch[0].shape) == len(tuple(batch_shape))
    assert len(batch[1].shape) == len(tuple(batch_shape))
    assert len(batch[2].shape) == len(tuple(batch_shape))

    placeholders = data_shuffler("data", from_queue=False)
    assert placeholders['anchor'].get_shape().as_list() == batch_shape
    assert placeholders['positive'].get_shape().as_list() == batch_shape
    assert placeholders['negative'].get_shape().as_list() == batch_shape


def test_diskaudio_shuffler():

    train_data, train_labels = get_dummy_audiofiles()

    batch_size = 582
    batch_shape = [None, 6560, 1]

    data_shuffler = DiskAudio(train_data, train_labels, batch_shape, batch_size=batch_size)
    batch = data_shuffler.get_batch()

    assert len(batch) == 2
    assert batch[0].shape[0] == batch_size
    assert batch[1].shape[0] == batch_size

    placeholders = data_shuffler("data", from_queue=False)
    assert placeholders.get_shape().as_list() == batch_shape
