from bob.extension.config import load
from collections import Iterable, Mapping


class PureDataTransform:
    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        context = None
        should_load = False

        if isinstance(transform, Mapping):
            transform, context = transform["transform"], transform.get("context")

        if isinstance(transform, str):
            transform = [transform]

        if isinstance(transform, Iterable):
            should_load = True

        if should_load:
            transform = load(
                transform,
                context=context,
                entry_point_group="bob.numpy.transform",
                attribute_name="transform",
            )

        self.transform = transform

    def __call__(self, inputs):
        inputs["data"] = self.transform(inputs["data"])
        return inputs
