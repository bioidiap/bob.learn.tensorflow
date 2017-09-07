import os
from bob.bio.base.test.dummy.database import database
preprocessor = extractor = algorithm = 'dummy'
groups = ['dev']

temp_directory = result_directory = 'TEST_DIR'
sub_directory = 'sub_directory'

data_dir = os.path.join('TEST_DIR', sub_directory, 'preprocessed')

# the directory to save the tfrecords in:
output_dir = os.path.join('TEST_DIR', sub_directory)

CLIENT_IDS = (str(f.client_id) for f in database.all_files(groups=groups))
CLIENT_IDS = list(set(CLIENT_IDS))
CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))


def file_to_label(f):
    return CLIENT_IDS[str(f.client_id)]
