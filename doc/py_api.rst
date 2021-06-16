.. vim: set fileencoding=utf-8 :

.. _bob.learn.tensorflow.py_api:

============
 Python API
============


Models
======

.. autosummary::
    bob.learn.tensorflow.models.AlexNet_simplified
    bob.learn.tensorflow.models.DeepPixBiS
    bob.learn.tensorflow.models.DenseNet
    bob.learn.tensorflow.models.densenet161
    bob.learn.tensorflow.models.MineModel

Data
====

.. autosummary::
    bob.learn.tensorflow.data.dataset_using_generator
    bob.learn.tensorflow.data.dataset_to_tfrecord
    bob.learn.tensorflow.data.dataset_from_tfrecord


Losses
======

.. autosummary::
    bob.learn.tensorflow.losses.CenterLossLayer
    bob.learn.tensorflow.losses.CenterLoss
    bob.learn.tensorflow.losses.PixelwiseBinaryCrossentropy
    bob.learn.tensorflow.losses.balanced_sigmoid_cross_entropy_loss_weights
    bob.learn.tensorflow.losses.balanced_softmax_cross_entropy_loss_weights


Image Utilities
===============

.. autosummary::
    bob.learn.tensorflow.utils.image.to_channels_last
    bob.learn.tensorflow.utils.image.to_channels_first
    bob.learn.tensorflow.utils.image.blocks_tensorflow
    bob.learn.tensorflow.utils.image.tf_repeat
    bob.learn.tensorflow.utils.image.all_patches


Keras Utilities
===============

.. autosummary::
    bob.learn.tensorflow.utils.keras.keras_channels_index
    bob.learn.tensorflow.utils.keras.keras_model_weights_as_initializers_for_variables
    bob.learn.tensorflow.utils.keras.restore_model_variables_from_checkpoint
    bob.learn.tensorflow.utils.keras.initialize_model_from_checkpoint
    bob.learn.tensorflow.utils.keras.model_summary


Math Utilities
==============

.. autosummary::
    bob.learn.tensorflow.utils.math.gram_matrix
    bob.learn.tensorflow.utils.math.upper_triangle_and_diagonal
    bob.learn.tensorflow.utils.math.upper_triangle
    bob.learn.tensorflow.utils.math.pdist
    bob.learn.tensorflow.utils.math.cdist
    bob.learn.tensorflow.utils.math.random_choice_no_replacement


Detailed Information
====================

.. automodule:: bob.learn.tensorflow
.. automodule:: bob.learn.tensorflow.data
.. automodule:: bob.learn.tensorflow.losses
.. automodule:: bob.learn.tensorflow.models
.. automodule:: bob.learn.tensorflow.utils
.. automodule:: bob.learn.tensorflow.utils.image
.. automodule:: bob.learn.tensorflow.utils.keras
.. automodule:: bob.learn.tensorflow.utils.math
