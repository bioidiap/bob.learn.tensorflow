import tensorflow as tf
from .Base import Base


class Estimator(Base):
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        kwargs['checkpoint'] = kwargs.get('checkpoint', estimator.model_dir)
        super().__init__(**kwargs)

    def get_output(self, data, mode):
        features = {'data': data, 'key': tf.constant(['key'])}
        self.estimator_spec = self.estimator._call_model_fn(
            features, None, mode, None)
        self.end_points = self.estimator.end_points
        return self.end_points[self.output_name]
