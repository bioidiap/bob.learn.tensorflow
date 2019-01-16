from bob.learn.tensorflow.dataset.bio import BioGenerator
from bob.learn.tensorflow.utils import to_channels_last
import tensorflow as tf

batch_size = 2
epochs = 2


def input_fn(mode):
    from bob.bio.base.test.dummy.database import database as db

    if mode == tf.estimator.ModeKeys.TRAIN:
        groups = 'world'
    elif mode == tf.estimator.ModeKeys.EVAL:
        groups = 'dev'

    files = db.objects(groups=groups)

    # construct integer labels for each identity in the database
    CLIENT_IDS = (str(f.client_id) for f in files)
    CLIENT_IDS = list(set(CLIENT_IDS))
    CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))

    def biofile_to_label(f):
        return CLIENT_IDS[str(f.client_id)]

    def load_data(database, f):
        img = f.load(database.original_directory, database.original_extension)
        # make a channels_first image (bob format) with 1 channel
        img = img.reshape(1, 112, 92)
        return img

    generator = BioGenerator(db, files, load_data, biofile_to_label)

    dataset = tf.data.Dataset.from_generator(
        generator, generator.output_types, generator.output_shapes)

    def transform(image, label, key):
        # convert to channels last
        image = to_channels_last(image)

        # per_image_standardization
        image = tf.image.per_image_standardization(image)
        return (image, label, key)

    dataset = dataset.map(transform)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # since we are caching to memory, caching only in training makes sense.
        dataset = dataset.cache()
        dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    data, label, key = dataset.make_one_shot_iterator().get_next()
    return {'data': data, 'key': key}, label


def train_input_fn():
    return input_fn(tf.estimator.ModeKeys.TRAIN)


def eval_input_fn():
    return input_fn(tf.estimator.ModeKeys.EVAL)


train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=50)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
