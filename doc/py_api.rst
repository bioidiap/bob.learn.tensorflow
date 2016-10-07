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
    bob.learn.tensorflow.network.VGG


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


Data Shufflers
--------------

.. autosummary::

    bob.learn.tensorflow.data.BaseDataShuffler
    bob.learn.tensorflow.data.MemoryDataShuffler
    bob.learn.tensorflow.data.TextDataShuffler


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
    bob.learn.tensorflow.initialization.SimpleXavier
    bob.learn.tensorflow.initialization.Xavier


Loss
----

.. autosummary::

    bob.learn.tensorflow.loss.BaseLoss
    bob.learn.tensorflow.loss.ConstrastiveLoss
    bob.learn.tensorflow.loss.TripletLoss

Detailed Information
--------------------

.. automodule:: bob.learn.tensorflow
.. automodule:: bob.learn.tensorflow.network
.. automodule:: bob.learn.tensorflow.trainers
.. automodule:: bob.learn.tensorflow.layers
.. automodule:: bob.learn.tensorflow.data
.. automodule:: bob.learn.tensorflow.network
.. automodule:: bob.learn.tensorflow.analyzers
.. automodule:: bob.learn.tensorflow.initialization
.. automodule:: bob.learn.tensorflow.loss