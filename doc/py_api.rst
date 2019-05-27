.. vim: set fileencoding=utf-8 :

.. _bob.learn.tensorflow.py_api:

============
 Python API
============


Estimators
==========

.. autosummary::
    bob.learn.tensorflow.estimators.Logits
    bob.learn.tensorflow.estimators.LogitsCenterLoss
    bob.learn.tensorflow.estimators.Triplet
    bob.learn.tensorflow.estimators.Siamese
    bob.learn.tensorflow.estimators.Regressor
    bob.learn.tensorflow.estimators.MovingAverageOptimizer
    bob.learn.tensorflow.estimators.learning_rate_decay_fn



Architectures
=============

.. autosummary::
    bob.learn.tensorflow.network.chopra
    bob.learn.tensorflow.network.light_cnn9
    bob.learn.tensorflow.network.dummy
    bob.learn.tensorflow.network.mlp
    bob.learn.tensorflow.network.mlp_with_batchnorm_and_dropout
    bob.learn.tensorflow.network.inception_resnet_v2
    bob.learn.tensorflow.network.inception_resnet_v1
    bob.learn.tensorflow.network.inception_resnet_v2_batch_norm
    bob.learn.tensorflow.network.inception_resnet_v1_batch_norm
    bob.learn.tensorflow.network.SimpleCNN.slim_architecture
    bob.learn.tensorflow.network.vgg_19
    bob.learn.tensorflow.network.vgg_16


Data
====

.. autosummary::
    bob.learn.tensorflow.dataset.bio.BioGenerator
    bob.learn.tensorflow.dataset.image.shuffle_data_and_labels_image_augmentation
    bob.learn.tensorflow.dataset.siamese_image.shuffle_data_and_labels_image_augmentation
    bob.learn.tensorflow.dataset.triplet_image.shuffle_data_and_labels_image_augmentation
    bob.learn.tensorflow.dataset.tfrecords.shuffle_data_and_labels_image_augmentation
    bob.learn.tensorflow.dataset.tfrecords.shuffle_data_and_labels
    bob.learn.tensorflow.dataset.generator.dataset_using_generator
    bob.learn.tensorflow.utils.util.to_channels_last
    bob.learn.tensorflow.utils.util.to_channels_first


Style Transfer
==============

.. autosummary::
    bob.learn.tensorflow.style_transfer.do_style_transfer


Losses
======

.. autosummary::
    bob.learn.tensorflow.loss.mean_cross_entropy_loss
    bob.learn.tensorflow.loss.mean_cross_entropy_center_loss
    bob.learn.tensorflow.loss.contrastive_loss
    bob.learn.tensorflow.loss.triplet_loss
    bob.learn.tensorflow.loss.triplet_average_loss
    bob.learn.tensorflow.loss.triplet_fisher_loss
    bob.learn.tensorflow.loss.linear_gram_style_loss
    bob.learn.tensorflow.loss.content_loss
    bob.learn.tensorflow.loss.denoising_loss
    bob.learn.tensorflow.loss.balanced_softmax_cross_entropy_loss_weights
    bob.learn.tensorflow.loss.balanced_sigmoid_cross_entropy_loss_weights




Detailed Information
====================

.. automodule:: bob.learn.tensorflow
.. automodule:: bob.learn.tensorflow.estimators
.. automodule:: bob.learn.tensorflow.dataset
.. automodule:: bob.learn.tensorflow.dataset.generator
.. automodule:: bob.learn.tensorflow.dataset.bio
.. automodule:: bob.learn.tensorflow.dataset.image
.. automodule:: bob.learn.tensorflow.dataset.siamese_image
.. automodule:: bob.learn.tensorflow.dataset.triplet_image
.. automodule:: bob.learn.tensorflow.dataset.tfrecords
.. automodule:: bob.learn.tensorflow.network
.. automodule:: bob.learn.tensorflow.network.SimpleCNN
.. automodule:: bob.learn.tensorflow.utils
.. automodule:: bob.learn.tensorflow.utils.util
.. automodule:: bob.learn.tensorflow.style_transfer
.. automodule:: bob.learn.tensorflow.loss
