import os
import six
import tensorflow as tf
from bob.bio.base.tools.grid import indices
from bob.bio.base import read_original_data as _read_original_data


def make_output_path(output_dir, key):
    return os.path.join(output_dir, key + '.hdf5')


def load_data(biofile, read_original_data, original_directory,
              original_extension):
    data = read_original_data(biofile, original_directory, original_extension)
    return data


def bio_generator(database, groups, number_of_parallel_jobs, output_dir,
                  read_original_data=None, biofile_to_label=None,
                  multiple_samples=False, force=False):
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
            data = load_data(f, read_original_data, database)
            # labels
            if multiple_samples:
                for d in data:
                    yield (d, label, key)
            else:
                yield (data, label, key)

    # load one data to get its type and shape
    data = load_data(biofiles[0], read_original_data, database)
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
