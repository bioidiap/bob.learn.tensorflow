import os
import six
import tensorflow as tf
from bob.bio.base.tools.grid import indices
from bob.bio.base import read_original_data as _read_original_data


def make_output_path(output_dir, key):
    """Returns an output path used for saving keys. You need to make sure the
    directories leading to this output path exist.

    Parameters
    ----------
    output_dir : str
        The root directory to save the results
    key : str
        The key of the sample. Usually biofile.make_path("", "")

    Returns
    -------
    str
        The path for the provided key.
    """
    return os.path.join(output_dir, key + '.hdf5')


def bio_generator(database, groups, number_of_parallel_jobs, output_dir,
                  read_original_data=None, biofile_to_label=None,
                  multiple_samples=False, force=False):
    """Returns a generator and its output types and shapes based on
    bob.bio.base databases.

    Parameters
    ----------
    database : :any:`bob.bio.base.database.BioDatabase`
        The database that you want to use.
    groups : [str]
        List of groups. Can be any permutation of ``('world', 'dev', 'eval')``
    number_of_parallel_jobs : int
        The number of parallel jobs that the script has ran with. This is used
        to split the number files into array jobs.
    output_dir : str
        The root directory where the data will be saved.
    read_original_data : :obj:`object`, optional
        A callable with the signature of
        ``data = read_original_data(biofile, directory, extension)``.
        :any:`bob.bio.base.read_original_data` is used by default.
    biofile_to_label : :obj:`object`, optional
        A callable with the signature of ``label = biofile_to_label(biofile)``.
        By default -1 is returned as label.
    multiple_samples : bool, optional
        If true, it assumes that the bio database's samples actually contain
        multiple samples. This is useful for when you want to treat video
        databases as image databases.
    force : bool, optional
        If true, all files will be overwritten.

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
    if read_original_data is None:
        read_original_data = _read_original_data
    if biofile_to_label is None:
        def biofile_to_label(biofile):
            return -1
    biofiles = list(database.all_files(groups))
    if number_of_parallel_jobs > 1:
        start, end = indices(biofiles, number_of_parallel_jobs)
        biofiles = biofiles[start:end]
    labels = (biofile_to_label(f) for f in biofiles)
    keys = (str(f.make_path("", "")) for f in biofiles)

    def generator():
        for f, label, key in six.moves.zip(biofiles, labels, keys):
            outpath = make_output_path(output_dir, key)
            if not force and os.path.isfile(outpath):
                continue
            data = read_original_data(f, database.original_directory,
                                      database.original_extension)
            # labels
            if multiple_samples:
                for d in data:
                    yield (d, label, key)
            else:
                yield (data, label, key)

    # load one data to get its type and shape
    data = read_original_data(biofiles[0], database.original_directory,
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
