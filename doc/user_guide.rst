.. vim: set fileencoding=utf-8 :
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.learn.tensorflow:

===========
 User guide
===========


Getting started
---------------

Before explain the base elements of this library, lets first do a simple example.
The example consists in training a very simple **CNN** with `MNIST` dataset in 4 steps.


1. Preparing your input data

.. doctest::

    >>> import tensorflow as tf
    >>> import bob.learn.tensorflow
    >>> import numpy

    >>> train_data, train_labels, _, _ = bob.learn.tensorflow.util.load_mnist()
    >>> train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    >>> train_data_shuffler = bob.learn.tensorflow.datashuffler.Memory(train_data, train_labels, input_shape=[28, 28, 1], batch_size=16)

2. Create an architecture

.. doctest::

    >>> architecture = bob.learn.tensorflow.network.SequenceNetwork()
    >>> architecture.add(bob.learn.tensorflow.layers.Conv2D(name="conv1", kernel_size=3, filters=10, activation=tf.nn.tanh))
    >>> architecture.add(bob.learn.tensorflow.layers.FullyConnected(name="fc1", output_dim=10, activation=None))

3. Defining a loss and training

.. doctest::

    >>> loss = bob.learn.tensorflow.loss.BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)
    >>> trainer = bob.learn.tensorflow.trainers.Trainer(architecture=architecture, loss=loss, iterations=100, temp_dir="./cnn")
    >>> trainer.train(train_data_shuffler)


4. Predicting and computing the accuracy

.. doctest::

    >>> with tf.Session() as session:
    >>>    # Loading the model
    >>>    architecture = bob.learn.tensorflow.network.SequenceNetwork()
    >>>    architecture.load(session, "./cnn/model.ckp")
    >>>    # Predicting
    >>>    predictions = scratch.predict(validation_data, session=session)
    >>>    # Computing an awesome accuracy for a simple network and 100 iterations
    >>>    accuracy = 100. * numpy.sum(predictions == validation_labels) / predictions.shape[0]
    >>>    print accuracy
    90.4714285714

Now lets describe each step in detail.


Preparing your input data
.........................

In this library datasets are wrapped in **data shufflers**. Data shufflers are elements designed to shuffle
the input data for stochastic training.
It has one basic functionality which is :py:meth:`bob.learn.tensorflow.datashuffler.Base.get_batch` functionality.

It is possible to either use Memory (:py:class:`bob.learn.tensorflow.datashuffler.Memory`) or
Disk (:py:class:`bob.learn.tensorflow.datashuffler.Disk`) data shufflers.

For the Memory data shufflers, as in the example, it is expected that the dataset is stored in `numpy.array`.
In the example that we provided the MNIST dataset was loaded and reshaped to `[n, w, h, c]` where `n` is the size
of the batch, `w` and `h` are the image width and height and `c` is the
number of channels.


Creating the architecture
.........................

Architectures are assembled as a :py:class:`bob.learn.tensorflow.network.SequenceNetwork` object.
Once the objects are created it is necessary to fill it up with `Layers`_.
The library has already some crafted networks implemented in `Architectures`_


Defining a loss and training
............................

The loss function can be defined by any set of tensorflow operations.
In our example, we used the `tf.nn.sparse_softmax_cross_entropy_with_logits` loss, but we also have some crafted
loss functions for Siamese :py:class`bob.learn.tensorflow.loss.ContrastiveLoss` and Triplet networks :py:class`bob.learn.tensorflow.loss.TripletLoss`.

The trainer is the real muscle here.
This element takes the inputs and trains the network.
As for the loss, we have specific trainers for Siamese (:py:class:`bob.learn.tensorflow.trainers.SiameseTrainer`) a
nd Triplet networks (:py:class:`bob.learn.tensorflow.trainers.TripletTrainer`).


Sandbox
-------

We have a sandbox of examples in a git repository `https://gitlab.idiap.ch/tiago.pereira/bob.learn.tensorflow_sandbox`_
The sandbox has some example of training:
 - MNIST with softmax
 - MNIST with Siamese Network
 - MNIST with Triplet Network
 - Face recognition with MOBIO database
 - Face recognition with CASIA WebFace database
