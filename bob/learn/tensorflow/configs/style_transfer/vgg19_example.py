"""
Example using VGG19
"""

from bob.learn.tensorflow.network import vgg_19
# --architecture
architecture = vgg_19


import numpy

# -- checkpoint-dir
# YOU CAN DOWNLOAD THE CHECKPOINTS FROM HERE 
# https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
checkpoint_dir = "/PATH-TO/vgg_19.ckpt"

# --style-end-points and -- content-end-points
content_end_points = ['vgg_19/conv4/conv4_2', 'vgg_19/conv5/conv5_2']
style_end_points = ['vgg_19/conv1/conv1_2', 
                    'vgg_19/conv2/conv2_1',
                    'vgg_19/conv3/conv3_1',
                    'vgg_19/conv4/conv4_1',
                    'vgg_19/conv5/conv5_1'
                    ]


scopes = {"vgg_19/":"vgg_19/"}

style_image_paths = ["/PATH/TO/vincent_van_gogh.jpg"]


# --preprocess-fn and --un-preprocess-fn
# Taken from VGG19
def mean_norm(tensor):
    return tensor - numpy.array([ 123.68 ,  116.779,  103.939])

def un_mean_norm(tensor):
    return tensor + numpy.array([ 123.68 ,  116.779,  103.939])

preprocess_fn = mean_norm

un_preprocess_fn = un_mean_norm

