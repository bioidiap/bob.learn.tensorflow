from bob.bio.base.test.dummy.database import database
from bob.bio.base.utils import read_original_data
from bob.learn.tensorflow.dataset.generator import dataset_using_generator

groups = ["dev"]

samples = database.all_files(groups=groups)

CLIENT_IDS = (str(f.client_id) for f in database.all_files(groups=groups))
CLIENT_IDS = list(set(CLIENT_IDS))
CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))


def file_to_label(f):
    return CLIENT_IDS[str(f.client_id)]


def reader(biofile):
    data = read_original_data(
        biofile, database.original_directory, database.original_extension
    )
    label = file_to_label(biofile)
    key = str(biofile.path).encode("utf-8")
    return (data, label, key)


dataset = dataset_using_generator(samples, reader)
datasets = [dataset]
