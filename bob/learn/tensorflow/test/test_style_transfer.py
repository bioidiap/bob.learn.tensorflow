from __future__ import print_function
import os
import shutil
from glob import glob
from tempfile import mkdtemp
from click.testing import CliRunner
from bob.io.base.test_utils import datafile
import pkg_resources

import tensorflow as tf

from bob.learn.tensorflow.utils import load_mnist, create_mnist_tfrecord
from bob.learn.tensorflow.utils.hooks import LoggerHookEstimator
from bob.learn.tensorflow.loss import mean_cross_entropy_loss
from bob.learn.tensorflow.utils import reproducible
from .test_estimator_onegraph import run_logitstrainer_mnist

from bob.learn.tensorflow.estimators import Logits
from bob.learn.tensorflow.network import dummy
from bob.learn.tensorflow.script.style_transfer import style_transfer
from nose.plugins.attrib import attr

dummy_config = datafile('style_transfer.py', __name__)
CONFIG = '''
from bob.learn.tensorflow.network import dummy
architecture = dummy
import pkg_resources

checkpoint_dir = "./temp/"

style_end_points = ["conv1"]
content_end_points = ["fc1"]

scopes = {"Dummy/":"Dummy/"}

'''


#tfrecord_train = "./train_mnist.tfrecord"
model_dir = "./temp"
output_style_image = 'output_style.png'

learning_rate = 0.1
data_shape = (28, 28, 1)  # size of atnt images
data_type = tf.float32
batch_size = 32
epochs = 1
steps = 100

# @attr('slow')
# def test_style_transfer():
#     with open(dummy_config, 'w') as f:
#         f.write(CONFIG)

#     # Trainer logits

#     # CREATING FAKE MODEL USING MNIST
#     _, run_config,_,_,_ = reproducible.set_seed()
#     trainer = Logits(
#         model_dir=model_dir,
#         architecture=dummy,
#         optimizer=tf.train.GradientDescentOptimizer(learning_rate),
#         n_classes=10,
#         loss_op=mean_cross_entropy_loss,
#         config=run_config)
#     run_logitstrainer_mnist(trainer)

#     # Style transfer using this fake model
#     runner = CliRunner()
#     result = runner.invoke(style_transfer,
#                            args=[pkg_resources.resource_filename( __name__, 'data/dummy_image_database/m301_01_p01_i0_0_GRAY.png'),
#                                output_style_image, dummy_config])

#     try:
#         os.unlink(dummy_config)
#         shutil.rmtree(model_dir, ignore_errors=True)
#     except Exception:
#         pass


