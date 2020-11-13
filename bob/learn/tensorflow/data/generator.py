import logging
import random

import tensorflow as tf

logger = logging.getLogger(__name__)


class Generator:
    """A generator class which wraps samples so that they can
    be used with tf.data.Dataset.from_generator

    Attributes
    ----------
    epoch : int
        The number of epochs that have been passed so far.

    multiple_samples : :obj:`bool`, optional
        If true, it assumes that the bio database's samples actually contain
        multiple samples. This is useful for when you want to for example treat
        video databases as image databases.

    reader : :obj:`object`, optional
        A callable with the signature of ``data, label, key = reader(sample)``
        which takes a sample and loads it.

    samples : [:obj:`object`]
        A list of samples to be given to ``reader`` to load the data.

    shuffle_on_epoch_end : :obj:`bool`, optional
        If True, it shuffle the samples at the end of each epoch.
    """

    def __init__(
        self,
        samples,
        reader,
        multiple_samples=False,
        shuffle_on_epoch_end=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reader = reader
        self.samples = list(samples)
        self.multiple_samples = multiple_samples
        self.epoch = 0
        self.shuffle_on_epoch_end = shuffle_on_epoch_end

        # load samples until one of them is not empty
        # this data is used to get the type and shape
        for sample in self.samples:
            try:
                dlk = self.reader(sample)
                if self.multiple_samples:
                    try:
                        dlk = dlk[0]
                    except TypeError:
                        # if the data is a generator
                        dlk = next(dlk)
            except StopIteration:
                continue
            else:
                break
        # Creating a "fake" dataset just to get the types and shapes
        dataset = tf.data.Dataset.from_tensors(dlk)
        self._output_types = tf.compat.v1.data.get_output_types(dataset)
        self._output_shapes = tf.compat.v1.data.get_output_shapes(dataset)

        logger.info(
            "Initializing a dataset with %d %s and %s types and %s shapes",
            len(self.samples),
            "multi-samples" if self.multiple_samples else "samples",
            self.output_types,
            self.output_shapes,
        )

    @property
    def output_types(self):
        "The types of the returned samples"
        return self._output_types

    @property
    def output_shapes(self):
        "The shapes of the returned samples"
        return self._output_shapes

    def __call__(self):
        """A generator function that when called will yield the samples.

        Yields
        ------
        object
            Samples one by one.
        """
        for sample in self.samples:
            dlk = self.reader(sample)
            if self.multiple_samples:
                for sub_dlk in dlk:
                    yield sub_dlk
            else:
                yield dlk
        self.epoch += 1
        logger.info("Elapsed %d epoch(s)", self.epoch)
        if self.shuffle_on_epoch_end:
            logger.info("Shuffling samples")
            random.shuffle(self.samples)


def dataset_using_generator(samples, reader, **kwargs):
    """
    A generator class which wraps samples so that they can
    be used with tf.data.Dataset.from_generator

    Parameters
    ----------
    samples : [:obj:`object`]
       A list of samples to be given to ``reader`` to load the data.

    reader : :obj:`object`, optional
       A callable with the signature of ``data, label, key = reader(sample)``
       which takes a sample and loads it.
    **kwargs
        Extra keyword arguments are passed to Generator

    Returns
    -------
    object
        A tf.data.Dataset
    """

    generator = Generator(samples, reader, **kwargs)
    dataset = tf.data.Dataset.from_generator(
        generator, generator.output_types, generator.output_shapes
    )
    return dataset
