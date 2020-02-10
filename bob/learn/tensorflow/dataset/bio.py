from bob.bio.base import read_original_data
from .generator import Generator
import logging

logger = logging.getLogger(__name__)


class BioGenerator(Generator):
    """A generator class which wraps bob.bio.base databases so that they can
    be used with tf.data.Dataset.from_generator

    Attributes
    ----------
    biofile_to_label : :obj:`object`, optional
        A callable with the signature of ``label = biofile_to_label(biofile)``.
        By default -1 is returned as label.
    database : :any:`bob.bio.base.database.BioDatabase`
        The database that you want to use.
    load_data : :obj:`object`, optional
        A callable with the signature of
        ``data = load_data(database, biofile)``.
        :any:`bob.bio.base.read_original_data` is wrapped to be used by
        default.
    biofiles : [:any:`bob.bio.base.database.BioFile`]
        The list of the bio files .
    keys : [str]
        The keys of samples obtained by calling ``biofile.make_path("", "")``
    labels : [int]
        The labels obtained by calling ``label = biofile_to_label(biofile)``
    """

    def __init__(
        self,
        database,
        biofiles,
        load_data=None,
        biofile_to_label=None,
        multiple_samples=False,
        **kwargs
    ):

        if load_data is None:

            def load_data(database, biofile):
                data = read_original_data(
                    biofile, database.original_directory, database.original_extension
                )
                return data

        if biofile_to_label is None:

            def biofile_to_label(biofile):
                return -1

        self.database = database
        self.load_data = load_data
        self.biofile_to_label = biofile_to_label

        def _reader(f):
            label = int(self.biofile_to_label(f))
            data = self.load_data(self.database, f)
            key = str(f.make_path("", "")).encode("utf-8")
            return data, label, key

        if multiple_samples:
            def reader(f):
                data, label, key = _reader(f)
                for d in data:
                    yield (d, label, key)
        else:
            def reader(f):
                return _reader(f)

        super(BioGenerator, self).__init__(
            biofiles, reader, multiple_samples=multiple_samples, **kwargs
        )

    @property
    def labels(self):
        for f in self.biofiles:
            yield int(self.biofile_to_label(f))

    @property
    def keys(self):
        for f in self.biofiles:
            yield str(f.make_path("", "")).encode("utf-8")

    @property
    def biofiles(self):
        return self.samples

    def __len__(self):
        return len(self.biofiles)
