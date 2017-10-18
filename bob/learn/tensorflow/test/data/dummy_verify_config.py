from bob.bio.base.test.dummy.database import database
from bob.bio.base.test.dummy.preprocessor import preprocessor

groups = 'dev'

files = database.all_files(groups=groups)

CLIENT_IDS = (str(f.client_id) for f in database.all_files(groups=groups))
CLIENT_IDS = list(set(CLIENT_IDS))
CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))


def file_to_label(f):
    return CLIENT_IDS[str(f.client_id)]


def reader(biofile):
    data = preprocessor.read_original_data(
        biofile, database.original_directory, database.original_extension)
    label = file_to_label(biofile)
    key = biofile.path
    return (data, label, key)
