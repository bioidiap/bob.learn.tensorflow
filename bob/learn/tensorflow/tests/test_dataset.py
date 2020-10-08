import numpy as np

from bob.learn.tensorflow.data import dataset_using_generator


def test_dataset_using_generator():
    def reader(f):
        key = 0
        label = 0
        yield {"data": f, "key": key}, label

    shape = (2, 2, 1)
    samples = [np.ones(shape, dtype="float32") * i for i in range(10)]

    dataset = dataset_using_generator(samples, reader, multiple_samples=True)
    for i, sample in enumerate(dataset):
        assert sample[0]["data"].shape == shape
        assert np.allclose(sample[0]["data"], samples[i])
