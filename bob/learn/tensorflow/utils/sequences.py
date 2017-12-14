from __future__ import division
import numpy
from keras.utils import Sequence
# documentation imports
from bob.dap.base.database import PadDatabase, PadFile
from bob.bio.base.preprocessor import Preprocessor


class PadSequence(Sequence):
    """A data shuffler for bob.dap.base database interfaces.

    Attributes
    ----------
    batch_size : int
        The number of samples to return in every batch.
    files : list of :any:`PadFile`
        List of file objects for a particular group and protocol.
    labels : list of bool
        List of labels for the files. ``True`` if bona-fide, ``False`` if
        attack.
    preprocessor : :any:`Preprocessor`
        The preprocessor to be used to load and process the data.
    """

    def __init__(self, files, labels, batch_size, preprocessor,
                 original_directory, original_extension):
        super(PadSequence, self).__init__()
        self.files = files
        self.labels = labels
        self.batch_size = int(batch_size)
        self.preprocessor = preprocessor
        self.original_directory = original_directory
        self.original_extension = original_extension

    def __len__(self):
        """Number of batch in the Sequence.

        Returns
        -------
        int
            The number of batches in the Sequence.
        """
        return int(numpy.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.load_batch(files, labels)

    def load_batch(self, files, labels):
        """Loads a batch of files and processes them.

        Parameters
        ----------
        files : list of :any:`PadFile`
            List of files to load.
        labels : list of bool
            List of labels corresponding to the files.

        Returns
        -------
        tuple of :any:`numpy.array`
            A tuple of (x, y): the data and their targets.
        """
        data, targets = [], []
        for file_object, target in zip(files, labels):
            loaded_data = self.preprocessor.read_original_data(
                file_object, self.original_directory, self.original_extension)
            preprocessed_data = self.preprocessor(loaded_data)
            data.append(preprocessed_data)
            targets.append(target)
        return numpy.array(data), numpy.array(targets)

    def on_epoch_end(self):
        pass


def shuffle_data(files, labels):
    indexes = numpy.arange(len(files))
    numpy.random.shuffle(indexes)
    return [files[i] for i in indexes], [labels[i] for i in indexes]


def get_pad_files_labels(database, groups):
    """Returns the pad files and their labels.

    Parameters
    ----------
    database : :any:`PadDatabase`
        The database to be used. The database should have a proper
        ``database.protocol`` attribute.
    groups : str
        The group to be used to return the data. One of ('world', 'dev',
        'eval'). 'world' means training data and 'dev' means validation data.

    Returns
    -------
    tuple
        A tuple of (files, labels) for that particular group and protocol.
    """
    files = database.samples(groups=groups, protocol=database.protocol)
    labels = ((f.attack_type is None) for f in files)
    labels = numpy.fromiter(labels, bool, len(files))
    return files, labels


def get_pad_sequences(database,
                      preprocessor,
                      batch_size,
                      groups=('world', 'dev', 'eval'),
                      shuffle=False,
                      limit=None):
    """Returns a list of :any:`Sequence` objects for the database.

    Parameters
    ----------
    database : :any:`PadDatabase`
        The database to be used. The database should have a proper
        ``database.protocol`` attribute.
    preprocessor : :any:`Preprocessor`
        The preprocessor to be used to load and process the data.
    batch_size : int
        The number of samples to return in every batch.
    groups : str
        The group to be used to return the data. One of ('world', 'dev',
        'eval'). 'world' means training data and 'dev' means validation data.

    Returns
    -------
    list of :any:`Sequence`
        The requested sequences to be used.
    """
    seqs = []
    for grp in groups:
        files, labels = get_pad_files_labels(database, grp)
        if shuffle:
            files, labels = shuffle_data(files, labels)
        if limit is not None:
            files, labels = files[:limit], labels[:limit]
        seqs.append(
            PadSequence(files, labels, batch_size, preprocessor,
                        database.original_directory,
                        database.original_extension))
    return seqs
