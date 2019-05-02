from .generic import Reader


class Frames(Reader):
    """A frames reader"""

    def __init__(self, **kwargs):
        super().__init__(multiple_samples=True, **kwargs)

    def call(self, inputs):
        f = inputs["db_smaple"]
        for frame in f.frames:
            inputs["features"]["data"] = frame
            yield inputs
