import tensorflow as tf
import collections
from bob.extension.config import load


class Reader:
    """The base class for all readers

    Attributes
    ----------
    database : object
        A bob database
    mode : str
        One of tf.estimator.ModeKeys
    multiple_samples : bool
        If False, the reader returns the loaded data. Otherwise, it will yeild
        multiple samples.
    """

    def __init__(self, database=None, mode=None, multiple_samples=False, **kwargs):
        super().__init__(**kwargs)
        self.database = database
        self.mode = mode
        self.multiple_samples = multiple_samples

    def call(self, inputs):
        """Loads (and transforms) the data.

        Parameters
        ----------
        inputs : dict
            A dictionary with at least the ``db_sample`` key.

        Returns
        -------
        outputs : dict
            Loaded the data in a dictionary. Put the features in the
            ``features`` key and labels in the ``labels`` key. If something
            goes wrong either return ``None`` to ignore the error or raise an
            Exception to stop.
        """
        raise NotImplementedError

    def __call__(self, db_sample):
        outputs = self.call({"db_sample": db_sample, "labels": None, "features": {}})

        if self.multiple_samples:
            for out in self.filter_nones(outputs):
                yield out["features"], out["labels"]
        else:
            return outputs["features"], outputs["labels"]

    def filter_nones(self, generator):
        return filter(lambda x: x is not None, generator)

    bio_groups = {
        tf.estimator.ModeKeys.TRAIN: ["world"],
        tf.estimator.ModeKeys.EVAL: ["dev"],
        tf.estimator.ModeKeys.PREDICT: None,
    }

    pad_groups = {
        tf.estimator.ModeKeys.TRAIN: ["train"],
        tf.estimator.ModeKeys.EVAL: ["dev"],
        tf.estimator.ModeKeys.PREDICT: None,
    }


class SequentialReader(Reader):
    """Constructs a reader composed of several readers

    Attributes
    ----------
    readers : list
        A list of reader classes to be initialized with database and mode.
    """

    def __init__(self, database, mode, readers, **kwargs):
        super().__init__(database=database, mode=mode, multiple_samples=True, **kwargs)
        self.readers = []
        for r in readers:
            if isinstance(r, collections.Iterable):
                r = load(r, entry_point_group="bob.db.reader", attribute_name="reader")
            self.readers.append(r(database=database, mode=mode))

    def call(self, inputs):
        def yield_sample(s):
            yield s

        generator = yield_sample(inputs)

        for r in self.readers:
            if r.multiple_samples:
                generator = (
                    subsample for inputs in generator for subsample in r.call(inputs)
                )
            else:
                generator = (r.call(inputs) for inputs in generator)
            generator = self.filter_nones(generator)

        return generator


def reader_from_data_transform(transform):
    """A function for converting simple data transforms to readers

    Parameters
    ----------
    transform : object
        Either a callable or a mapping (dict) with keys of
        ``tf.estimator.ModeKeys``.

    Returns
    -------
    object
        The generated reader class
    """

    if isinstance(transform, collections.Iterable) and not isinstance(
        transform, collections.Mapping
    ):
        transform = load(
            transform,
            entry_point_group="bob.numpy.transform",
            attribute_name="transform",
        )

    if isinstance(transform, collections.Mapping):
        for k, v in transform.items():
            if isinstance(v, collections.Iterable):
                transform[k] = load(
                    v,
                    entry_point_group="bob.numpy.transform",
                    attribute_name="transform",
                )

    class DataTransformReader(Reader):
        """A class for converting pure functions to readers."""

        def call(self, inputs):

            t = transform
            if isinstance(transform, collections.Mapping):
                t = transform[self.mode]

            inputs["features"]["data"] = t(inputs["features"]["data"])
            return inputs

    return DataTransformReader
