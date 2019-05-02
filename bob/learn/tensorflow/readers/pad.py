import random
import tensorflow as tf
from .generic import Reader


class DropPASamples(Reader):
    def __init__(self, keep_pa_samples, modes=None, **kwargs):
        super().__init__(**kwargs)
        self.keep_pa_samples = keep_pa_samples
        if modes is None:
            modes = (tf.estimator.ModeKeys.TRAIN,)
        self.modes = modes

    def call(self, inputs):
        if self.mode not in self.modes:
            return inputs

        if inputs["db_smaple"].attack_type is None:
            return inputs

        if random.random() < self.keep_pa_samples:
            return inputs
        else:
            return None


class DropBFSamples(Reader):
    def __init__(self, keep_bf_samples, modes=None, **kwargs):
        super().__init__(**kwargs)
        self.keep_bf_samples = keep_bf_samples
        if modes is None:
            modes = (tf.estimator.ModeKeys.TRAIN,)
        self.modes = modes

    def call(self, inputs):
        if self.mode not in self.modes:
            return inputs

        if inputs["db_smaple"].attack_type is not None:
            return inputs

        if random.random() < self.keep_bf_samples:
            return inputs
        else:
            return None
