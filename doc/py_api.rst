.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <laurent.el-shafey@idiap.ch>
.. Tue 28 Aug 2012 18:09:40 CEST

.. _py_api:

============
 Python API
============


Architectures
-------------

.. autosummary::

    bob.learn.tensorflow.network.Chopra
    bob.learn.tensorflow.network.Dummy
    bob.learn.tensorflow.network.MLP


Trainers
--------

.. autosummary::

    bob.learn.tensorflow.trainers.Trainer
    bob.learn.tensorflow.trainers.SiameseTrainer
    bob.learn.tensorflow.trainers.TripletTrainer

Learning rate
-------------

.. autosummary::

    bob.learn.tensorflow.trainers.constant
    bob.learn.tensorflow.trainers.exponential_decay



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
    bob.learn.tensorflow.datashuffler.OnlineSampling



Data Augmentation
-----------------

.. autosummary::

    bob.learn.tensorflow.datashuffler.DataAugmentation
    bob.learn.tensorflow.datashuffler.ImageAugmentation


Analizers
---------

.. autosummary::

    bob.learn.tensorflow.analyzers.ExperimentAnalizer


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
.. automodule:: bob.learn.tensorflow.analyzers
.. automodule:: bob.learn.tensorflow.loss