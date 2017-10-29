import six
import tensorflow as tf
from bob.bio.base import read_original_data


def bio_generator(database, biofiles, load_data=None, biofile_to_label=None,
                  multiple_samples=False):
    """Returns a generator and its output types and shapes based on
    bob.bio.base databases.

    Parameters
    ----------
    database : :any:`bob.bio.base.database.BioDatabase`
        The database that you want to use.
    biofiles : [:any:`bob.bio.base.database.BioFile`]
        The list of the bio files .
    load_data : :obj:`object`, optional
        A callable with the signature of
        ``data = load_data(database, biofile)``.
        :any:`bob.bio.base.read_original_data` is used by default.
    biofile_to_label : :obj:`object`, optional
        A callable with the signature of ``label = biofile_to_label(biofile)``.
        By default -1 is returned as label.
    multiple_samples : bool, optional
        If true, it assumes that the bio database's samples actually contain
        multiple samples. This is useful for when you want to treat video
        databases as image databases.

    Returns
    -------
    generator : object
        A generator function that when called will return the samples. The
        samples will be like ``(data, label, key)``.
    output_types : (object, object, object)
        The types of the returned samples.
    output_shapes : (tf.TensorShape, tf.TensorShape, tf.TensorShape)
        The shapes of the returned samples.
    """
    if load_data is None:
        def load_data(database, biofile):
            data = read_original_data(
                biofile,
                database.original_directory,
                database.original_extension)
            return data
    if biofile_to_label is None:
        def biofile_to_label(biofile):
            return -1
    labels = (biofile_to_label(f) for f in biofiles)
    keys = (str(f.make_path("", "")) for f in biofiles)

    def generator():
        for f, label, key in six.moves.zip(biofiles, labels, keys):
            data = load_data(database, f)
            # labels
            if multiple_samples:
                for d in data:
                    yield (d, label, key)
            else:
                yield (data, label, key)

    # load one data to get its type and shape
    data = load_data(biofiles[0], database.original_directory,
                     database.original_extension)
    if multiple_samples:
        try:
            data = data[0]
        except TypeError:
            # if the data is a generator
            data = six.next(data)
    data = tf.convert_to_tensor(data)
    output_types = (data.dtype, tf.int64, tf.string)
    output_shapes = (data.shape, tf.TensorShape([]), tf.TensorShape([]))

    return (generator, output_types, output_shapes)
