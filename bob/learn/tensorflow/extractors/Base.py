import tensorflow as tf
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_checkpoint_path(path):
    if os.path.splitext(path)[1] == ".meta":
        filename = os.path.splitext(path)[0]
    elif os.path.isdir(path):
        filename = tf.train.latest_checkpoint(path)
    else:
        filename = path

    return filename


class Base:
    def __init__(self, output_name, input_shape, checkpoint, scopes,
                 input_transform=None, output_transform=None,
                 input_dtype='float32', extra_feed=None, **kwargs):

        self.output_name = output_name
        self.input_shape = input_shape
        self.checkpoint = normalize_checkpoint_path(checkpoint)
        self.scopes = scopes
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.input_dtype = input_dtype
        self.extra_feed = extra_feed
        self.session = None
        super().__init__(**kwargs)

    def load(self):
        self.session = tf.Session(graph=tf.Graph())

        with self.session.as_default(), self.session.graph.as_default():

            self.input = data = tf.placeholder(self.input_dtype, self.input_shape)

            if self.input_transform is not None:
                data = self.input_transform(data)

            self.output = self.get_output(data, tf.estimator.ModeKeys.PREDICT)

            if self.output_transform is not None:
                self.output = self.output_transform(self.output)

            tf.train.init_from_checkpoint(
                ckpt_dir_or_file=self.checkpoint,
                assignment_map=self.scopes,
            )
            # global_variables_initializer must run after init_from_checkpoint
            self.session.run(tf.global_variables_initializer())
            logger.info('Restored the model from %s', self.checkpoint)

    def __call__(self, data):
        if self.session is None:
            self.load()

        data = np.ascontiguousarray(data, dtype=self.input_dtype)
        feed_dict = {self.input: data}
        if self.extra_feed is not None:
            feed_dict.update(self.extra_feed)

        return self.session.run(self.output, feed_dict=feed_dict)

    def get_output(self, data, mode):
        raise NotImplementedError()
