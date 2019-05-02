from bob.extension.config import load
import collections


class Transform:
    """Base class for all tensorflow transforms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, features, labels):
        return features, labels


class DataTransform(Transform):
    """A wrapper for pure data transforms"""

    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        if isinstance(transform, collections.Iterable):
            transform = load(
                transform,
                entry_point_group="bob.tensorflow.transform",
                attribute_name="transform",
            )
        self.transform = transform

    def __call__(self, features, labels):
        features["data"] = self.transform(features["data"])
        return features, labels


class SequentialTransform:
    """Aggregates several transforms as one transform sequentially."""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = []
        for t in transforms:
            if not isinstance(t, Transform):
                t = DataTransform(t)
            self.transforms.append(t)

    def __call__(self, features, labels):
        for t in self.transforms:
            features, labels = t(features, labels)
        return features, labels
