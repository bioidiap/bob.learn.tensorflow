import six
import tensorflow as tf
import logging

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
    output_types : (object, object, object)
        The types of the returned samples.
    output_shapes : ``(tf.TensorShape, tf.TensorShape, tf.TensorShape)``
        The shapes of the returned samples.
    """

    def __init__(self, samples, reader, multiple_samples=False, **kwargs):
        super().__init__(**kwargs)
        self.reader = reader
        self.samples = list(samples)
        self.multiple_samples = multiple_samples
        self.epoch = 0

        # load one data to get its type and shape
        dlk = self.reader(self.samples[0])
        if self.multiple_samples:
            try:
                dlk = dlk[0]
            except TypeError:
                # if the data is a generator
                dlk = six.next(dlk)
        # Creating a "fake" dataset just to get the types and shapes
        dataset = tf.data.Dataset.from_tensors(dlk)
        self._output_types = dataset.output_types
        self._output_shapes = dataset.output_shapes

        logger.info(
            "Initializing a dataset with %d %s and %s types and %s shapes",
            len(self.samples),
            "multi-samples" if self.multiple_samples else "samples",
            self.output_types,
            self.output_shapes,
        )

    @property
    def output_types(self):
        return self._output_types

    @property
    def output_shapes(self):
        return self._output_shapes

    def __call__(self):
        """A generator function that when called will yield the samples.

        Yields
        ------
        (data, label, key) : tuple
            A tuple containing the data, label, and the key.
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


def dataset_using_generator(*args, **kwargs):
    """
    A generator class which wraps samples so that they can
    be used with tf.data.Dataset.from_generator

    Attributes
    ----------

     samples : [:obj:`object`]
        A list of samples to be given to ``reader`` to load the data.

     reader : :obj:`object`, optional
        A callable with the signature of ``data, label, key = reader(sample)``
        which takes a sample and loads it.

     multiple_samples : :obj:`bool`, optional
        If true, it assumes that the bio database's samples actually contain
        multiple samples. This is useful for when you want to for example treat
        video databases as image databases.
     
    """

    generator = Generator(*args, **kwargs)
    dataset = tf.data.Dataset.from_generator(
        generator, generator.output_types, generator.output_shapes
    )
    return dataset
