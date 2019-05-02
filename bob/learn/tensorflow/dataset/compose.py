import tensorflow as tf
import random
import collections
from ..readers.generic import SequentialReader
from ..tf_transforms.generic import SequentialTransform
from .generator import Generator


def compose_dataset(database, mode, readers, tf_transforms, mode_to_groups="bio"):
    if mode_to_groups == "bio":
        mode_to_groups = SequentialReader.bio_groups
    elif mode_to_groups == "pad":
        mode_to_groups = SequentialReader.pad_groups

    reader = SequentialReader(database=database, mode=mode, readers=readers)
    samples = database.objects(groups=mode_to_groups[mode], protocol=database.protocol)
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        random.shuffle(samples)

    generator = Generator(samples, reader, multiple_samples=reader.multiple_samples)

    dataset = tf.data.Dataset.from_generator(
        generator, generator.output_types, generator.output_shapes
    )

    if isinstance(tf_transforms, collections.Mapping):
        tf_transforms = (
            tf_transforms.get("begin", [])
            + tf_transforms.get(mode, [])
            + tf_transforms.get("end", [])
        )

    if isinstance(tf_transforms, collections.Iterable):
        tf_transforms = SequentialTransform(tf_transforms)

    return dataset.map(tf_transforms)
