"""
Example using inception resnet v1

"""

import tensorflow as tf

# -- architecture
from bob.learn.tensorflow.network import inception_resnet_v1_batch_norm
architecture = inception_resnet_v1_batch_norm

# --checkpoint-dir
from bob.extension import rc
checkpoint_dir = rc['bob.bio.face_ongoing.inception-v1_batchnorm_rgb']

# --style-end-points and -- content-end-points
style_end_points = ["Conv2d_1a_3x3", "Conv2d_2b_3x3"]
content_end_points = ["Block8"]

scopes = {"InceptionResnetV1/":"InceptionResnetV1/"}

# --style-image-paths
style_image_paths = ["vincent_van_gogh.jpg",
                     "vincent_van_gogh2.jpg"]

# --preprocess-fn
preprocess_fn = tf.image.per_image_standardization
