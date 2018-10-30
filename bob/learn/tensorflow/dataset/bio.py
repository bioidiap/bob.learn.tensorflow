import six
import tensorflow as tf
from bob.bio.base import read_original_data
import logging

logger = logging.getLogger(__name__)


class BioGenerator(object):
    """A generator class which wraps bob.bio.base databases so that they can
    be used with tf.data.Dataset.from_generator

    Attributes
    ----------
    biofile_to_label : :obj:`object`, optional
        A callable with the signature of ``label = biofile_to_label(biofile)``.
        By default -1 is returned as label.
    biofiles : [:any:`bob.bio.base.database.BioFile`]
        The list of the bio files .
    database : :any:`bob.bio.base.database.BioDatabase`
        The database that you want to use.
    epoch : int
        The number of epochs that have been passed so far.
    keys : [str]
        The keys of samples obtained by calling ``biofile.make_path("", "")``
    labels : [int]
        The labels obtained by calling ``label = biofile_to_label(biofile)``
    load_data : :obj:`object`, optional
        A callable with the signature of
        ``data = load_data(database, biofile)``.
        :any:`bob.bio.base.read_original_data` is wrapped to be used by
        default.
    multiple_samples : :obj:`bool`, optional
        If true, it assumes that the bio database's samples actually contain
        multiple samples. This is useful for when you want to for example treat
        video databases as image databases.
    output_types : (object, object, object)
        The types of the returned samples.
    output_shapes : ``(tf.TensorShape, tf.TensorShape, tf.TensorShape)``
        The shapes of the returned samples.
    """

    def __init__(self,
                 database,
                 biofiles,
                 load_data=None,
                 biofile_to_label=None,
                 multiple_samples=False,
                 **kwargs):
        super(BioGenerator, self).__init__(**kwargs)
        if load_data is None:

            def load_data(database, biofile):
                data = read_original_data(biofile, database.original_directory,
                                          database.original_extension)
                return data

        if biofile_to_label is None:

            def biofile_to_label(biofile):
                return -1

        self.database = database
        self.biofiles = list(biofiles)
        self.load_data = load_data
        self.biofile_to_label = biofile_to_label
        self.multiple_samples = multiple_samples
        self.epoch = 0

        # load one data to get its type and shape
        data = load_data(database, biofiles[0])
        if multiple_samples:
            try:
                data = data[0]
            except TypeError:
                # if the data is a generator
                data = six.next(data)
        data = tf.convert_to_tensor(data)
        self._output_types = (data.dtype, tf.int64, tf.string)
        self._output_shapes = (data.shape, tf.TensorShape([]),
                               tf.TensorShape([]))

        logger.info(
            "Initializing a dataset with %d files and %s types "
            "and %s shapes", len(self.biofiles), self.output_types,
            self.output_shapes)

    @property
    def labels(self):
        for f in self.biofiles:
            yield int(self.biofile_to_label(f))

    @property
    def keys(self):
        for f in self.biofiles:
            yield str(f.make_path("", "")).encode('utf-8')

    @property
    def output_types(self):
        return self._output_types

    @property
    def output_shapes(self):
        return self._output_shapes

    def __len__(self):
        return len(self.biofiles)

    def __call__(self):
        """A generator function that when called will return the samples.

        Yields
        ------
        (data, label, key) : tuple
            A tuple containing the data, label, and the key.
        """
        for f, label, key in six.moves.zip(self.biofiles, self.labels,
                                           self.keys):
            data = self.load_data(self.database, f)
            if self.multiple_samples:
                for d in data:
                    yield (d, label, key)
            else:
                yield (data, label, key)
        self.epoch += 1
        logger.info("Elapsed %d epoch(s)", self.epoch)
