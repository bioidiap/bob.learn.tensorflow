# Required objects:

# you need a database object that inherits from
# bob.bio.base.database.BioDatabase (PAD dbs work too)
database = Database()

# the directory pointing to where the processed data is:
data_dir = '/idiap/temp/user/database_name/sub_directory/preprocessed'

# the directory to save the tfrecords in:
output_dir = '/idiap/temp/user/database_name/sub_directory'


# A function that converts a BioFile or a PadFile to a label:
# Example for PAD
def file_to_label(f):
    return f.attack_type is None


# Example for Bio (You may want to run this script for groups=['world'] only
# in biometric recognition experiments.)
CLIENT_IDS = (str(f.client_id) for f in db.all_files(groups=groups))
CLIENT_IDS = list(set(CLIENT_IDS))
CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))


def file_to_label(f):
    return CLIENT_IDS[str(f.client_id)]


# Optional objects:

# The groups that you want to create tfrecords for. It should be a list of
# 'world' ('train' in bob.pad.base), 'dev', and 'eval' values. [default:
# 'world']
groups = ['world']

# you need a reader function that reads the preprocessed files. [default:
# bob.bio.base.utils.load]
reader = Preprocessor().read_data
reader = Extractor().read_feature
# or
from bob.bio.base.utils import load as reader

# or a reader that casts images to uint8:


def reader(path):
    data = bob.bio.base.utils.load(path)
    return data.astype("uint8")


# extension of the preprocessed files. [default: '.hdf5']
data_extension = '.hdf5'

# Shuffle the files before writing them into a tfrecords. [default: False]
shuffle = True

# Whether the each file contains one sample or more. [default: True] If
# this is False, the loaded samples from a file are iterated over and each
# of them is saved as an independent feature.
one_file_one_sample = True
