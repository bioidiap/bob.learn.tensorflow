import h5py


class MeanNormalizer:

    """An HDF5 normalizer class. It reads the mean from an hdf5 dataset and
    normalizer the values based on the mean value.

    Attributes
    ----------
    mean : array_like
        A mean value read from the hdf5 file.
    """

    def __init__(self, hdf5):
        with h5py.File(hdf5, "r") as f:
            self.mean = f["/mean"]["data_mean"].value

    def __call__(self, data):
        return data - self.mean
