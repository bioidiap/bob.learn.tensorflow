import tensorflow as tf
from bob.bio.base.test.dummy.database import database
biofiles = database.all_files(['dev'])


def bio_predict_input_fn(generator, output_types, output_shapes):
    def input_fn():
        dataset = tf.data.Dataset.from_generator(
            generator, output_types, output_shapes)
        # apply all kinds of transformations here, process the data
        # even further if you want.
        dataset = dataset.prefetch(1)
        dataset = dataset.batch(10**3)
        images, labels, keys = dataset.make_one_shot_iterator().get_next()

        return {'data': images, 'key': keys}, labels
    return input_fn
