.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <laurent.el-shafey@idiap.ch>
.. Tue 28 Aug 2012 18:09:40 CEST

============
 Python API
============


Architectures
-------------

.. autosummary::

    bob.learn.tensorflow.network.SequenceNetwork
    bob.learn.tensorflow.network.Chopra
    bob.learn.tensorflow.network.Dummy
    bob.learn.tensorflow.network.Lenet
    bob.learn.tensorflow.network.LenetDropout
    bob.learn.tensorflow.network.MLP
    bob.learn.tensorflow.network.VGG16
    bob.learn.tensorflow.network.VGG16_mod


Trainers
--------

.. autosummary::

    bob.learn.tensorflow.trainers.Trainer
    bob.learn.tensorflow.trainers.SiameseTrainer
    bob.learn.tensorflow.trainers.TripletTrainer

Layers
------

.. autosummary::

    bob.learn.tensorflow.layers.Layer
    bob.learn.tensorflow.layers.Conv2D
    bob.learn.tensorflow.layers.Dropout
    bob.learn.tensorflow.layers.FullyConnected
    bob.learn.tensorflow.layers.MaxPooling
    bob.learn.tensorflow.layers.AveragePooling


Data Shufflers
--------------

.. autosummary::

    bob.learn.tensorflow.datashuffler.Base
    bob.learn.tensorflow.datashuffler.Memory
    bob.learn.tensorflow.datashuffler.Disk
    bob.learn.tensorflow.datashuffler.Siamese
    bob.learn.tensorflow.datashuffler.SiameseDisk
    bob.learn.tensorflow.datashuffler.SiameseMemory
    bob.learn.tensorflow.datashuffler.Triplet
    bob.learn.tensorflow.datashuffler.TripletDisk
    bob.learn.tensorflow.datashuffler.TripletMemory
    bob.learn.tensorflow.datashuffler.TripletWithFastSelectionDisk
    bob.learn.tensorflow.datashuffler.TripletWithSelectionDisk
    bob.learn.tensorflow.datashuffler.OnLineSampling



Data Augmentation
-----------------

.. autosummary::

    bob.learn.tensorflow.datashuffler.DataAugmentation
    bob.learn.tensorflow.datashuffler.ImageAugmentation


Analizers
---------

.. autosummary::

    bob.learn.tensorflow.analyzers.ExperimentAnalizer

Initialization
--------------

.. autosummary::

    bob.learn.tensorflow.initialization.Initialization
    bob.learn.tensorflow.initialization.Constant
    bob.learn.tensorflow.initialization.Gaussian
    bob.learn.tensorflow.initialization.SimplerXavier
    bob.learn.tensorflow.initialization.Xavier


Loss
----

.. autosummary::

    bob.learn.tensorflow.loss.BaseLoss
    bob.learn.tensorflow.loss.ContrastiveLoss
    bob.learn.tensorflow.loss.TripletLoss

Detailed Information
--------------------

.. automodule:: bob.learn.tensorflow
.. automodule:: bob.learn.tensorflow.network
.. automodule:: bob.learn.tensorflow.trainers
.. automodule:: bob.learn.tensorflow.layers
.. automodule:: bob.learn.tensorflow.datashuffler
.. automodule:: bob.learn.tensorflow.network
.. automodule:: bob.learn.tensorflow.analyzers
.. automodule:: bob.learn.tensorflow.initialization
.. automodule:: bob.learn.tensorflow.loss