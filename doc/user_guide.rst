.. vim: set fileencoding=utf-8 :
.. date: Thu Sep 20 11:58:57 CEST 2012


===========
 User guide
===========


Quick start
-----------

Before explain the base elements of this library, lets first do a simple example.
The example consists in training a very simple **CNN** with `MNIST` dataset in 4 steps.

1. Preparing your input data

.. code-block:: python

    >>> import tensorflow as tf
    >>> from bob.learn.tensorflow.loss import BaseLoss
    >>> from bob.learn.tensorflow.trainers import Trainer, constant
    >>> from bob.learn.tensorflow.utils import load_mnist
    >>> from bob.learn.tensorflow.datashuffler import Memory
    >>> from bob.learn.tensorflow.network import Embedding
    >>> import numpy

    >>> # Loading raw data
    >>> train_data, train_labels, _, _ = bob.learn.tensorflow.util.load_mnist()
    >>> train_data = numpy.reshape(train_data, (train_data.shape[0], 28, 28, 1))

    >>> # Preparing the datashuffler (our batching engine)
    >>> train_data_shuffler = Memory(train_data, train_labels, input_shape=[None, 28, 28, 1], batch_size=16)
    >>>

2. Create an architecture

.. code-block:: python

    >>> # Create a function or a class with the graph definition in the `__call__` method
    >>> def create_dummy_architecture(placeholder):
    >>>     initializer = tf.contrib.layers.xavier_initializer(seed=10)  # Weights initializer
    >>>     slim = tf.contrib.slim
    >>>     graph = slim.conv2d(placeholder, 10, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='conv1', weights_initializer=initializer)
    >>>     graph = slim.flatten(graph, scope='flatten1')
    >>>     graph = slim.fully_connected(graph, 10, activation_fn=None, scope='fc1', weights_initializer=initializer)
    >>>     return graph
    >>>


3. Defining a loss and training algorithm

.. code-block:: python

    >>> architecture = create_dummy_architecture
    >>>
    >>> loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)
    >>>
    >>> optimizer = tf.train.GradientDescentOptimizer(0.001)
    >>>
    >>> learning_rate = constant(base_learning_rate=0.001)
    >>>
    >>> trainer = Trainer

Now that you have defined your data, architecture, loss and training algorithm you can save this in a python file,
let's say `softmax.py`, and run:

.. code-block:: shell

    >>> ./bin/train.py softmax.py --iterations 100 --output-dir ./my_first_net


4. Predicting and computing the accuracy

Run the following code to evalutate the network that was just trained.

.. code-block:: python

    >>> # Loading the trained model
    >>> trainer = Trainer(train_data_shuffler)
    >>> directory = "./my_first_net/"
    >>> trainer.create_network_from_file(os.path.join(directory, "model.ckp"))

    >>> # Prediction
    >>> embedding = Embedding(trainer.data_ph, trainer.graph)
    >>> [data, labels] = train_data_shuffler.get_batch()
    >>> predictions = embedding(data)
    >>> accuracy = 100. * numpy.sum(numpy.argmax(predictions, axis=1) == labels) / predictions.shape[0]
    87.5


Understanding what you have done
--------------------------------


Preparing your input data
.........................

In this library datasets are wrapped in **data shufflers**. Data shufflers are elements designed to do batching.
It has one basic functionality which is :py:meth:`bob.learn.tensorflow.datashuffler.Base.get_batch` functionality.

It is possible to either use Memory (:py:class:`bob.learn.tensorflow.datashuffler.Memory`) or
Disk (:py:class:`bob.learn.tensorflow.datashuffler.Disk`) data shufflers.

For the Memory data shufflers, as in the example, it is expected that the dataset is stored in `numpy.array`.
In the example that we provided the MNIST dataset was loaded and reshaped to `[n, w, h, c]` where `n` is the size
of the batch, `w` and `h` are the image width and height and `c` is the
number of channels.


Creating the architecture
.........................

Architectures are assembled using the Tensorflow graphs.
There are plenty of ways to doing it; you can either use the `tensorflow <https://www.tensorflow.org/api_docs/python/tf/Graph>`_ API directly
or use one of the several available contribs such as `tf-slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`_,
`TFLearn <http://tflearn.org/>`_, etc...


Defining a loss and training
............................

The loss function can be defined by any set of tensorflow operations.
In our example, we used the `tf.nn.sparse_softmax_cross_entropy_with_logits` as loss function, but we also have some crafted
loss functions for Siamese :py:class:`bob.learn.tensorflow.loss.ContrastiveLoss` and Triplet networks :py:class:`bob.learn.tensorflow.loss.TripletLoss`.

The trainer is the real muscle here.
This element takes the inputs and trains the network.
As for the loss, we have specific trainers for Siamese (:py:class:`bob.learn.tensorflow.trainers.SiameseTrainer`) a
nd Triplet networks (:py:class:`bob.learn.tensorflow.trainers.TripletTrainer`).


Components in detail
--------------------

If you have reached this point it means that you want to understand a little bit more on how this library works.
The next sections give some details of each element.

Data Shufflers and trainers
...........................

As mentioned before, datasets are wrapped in **data shufflers**.
Data shufflers were designed to shuffle the input data for stochastic training.
It has one basic functionality which is :py:meth:`bob.learn.tensorflow.datashuffler.Base.get_batch` functionality.

The shufflers are categorized with respect to:
 1. How the data is fetched
 2. The type of the trainer
 3. How the data is sampled

How do you want to fetch your data?
```````````````````````````````````

The data can be fetched either from the memory (:py:class:`bob.learn.tensorflow.datashuffler.Memory`), as in out example, or from
disk (:py:class:`bob.learn.tensorflow.datashuffler.Disk`).
To train networks fetched from the disk, your training data must be a list of paths like in the example below:

.. code-block:: python

    >>> train_data = ['./file/id1_0.jpg', './file/id1_1.jpg', './file/id2_1.jpg']
    >>> train_labels = [0, 0, 1]

With disk data shufflers, the data is loaded on the fly.


Type of the trainer?
````````````````````

Here we have one data shuffler for each type of the trainer.

You will see in the section `Trainers <py_api.html#trainers>`_ that we have three types of trainer.
The first one is the regular trainer, which deals with one graph only (for example, if you training a network with
a softmax loss).
The data shuflers for this type of trainer must be a direct instance of either :py:class:`bob.learn.tensorflow.datashuffler.Memory`
or :py:class:`bob.learn.tensorflow.datashuffler.Disk`.

The second one is the :py:class:`bob.learn.tensorflow.trainers.SiameseTrainer` trainer, which is designed to train Siamese networks.
The data shuflers for this type of trainer must be a direct instance of either :py:class:`bob.learn.tensorflow.datashuffler.SiameseDisk` or
:py:class:`bob.learn.tensorflow.datashuffler.SiameseMemory`.

The third one is the :py:class:`bob.learn.tensorflow.trainers.TripletTrainer` trainer, which is designed to train Triplet networks.
The data shuflers for this type of trainer must be a direct instance of either :py:class:`bob.learn.tensorflow.datashuffler.TripletDisk`,
:py:class:`bob.learn.tensorflow.datashuffler.TripletMemory`, :py:class:`bob.learn.tensorflow.datashuffler.TripletWithFastSelectionDisk`
or :py:class:`bob.learn.tensorflow.datashuffler.TripletWithSelectionDisk`.


How the data is sampled ?
`````````````````````````

The paper [facenet_2015]_ introduced a new strategy to select triplets to train triplet networks (this is better described
here :py:class:`bob.learn.tensorflow.datashuffler.TripletWithSelectionDisk` and :py:class:`bob.learn.tensorflow.datashuffler.TripletWithFastSelectionDisk`).
This triplet selection relies in the current state of the network and are extensions of `bob.learn.tensorflow.datashuffler.OnlineSampling`.


Activations
...........

For the activation of the layers we don't have any special wrapper.
For any class that inherits from :py:class:`bob.learn.tensorflow.layers.Layer` you can use directly tensorflow operations
in the keyword argument `activation`.


Solvers/Optimizer
.................

For the solvers we don't have any special wrapper.
For any class that inherits from :py:class:`bob.learn.tensorflow.trainers.Trainer` you can use directly tensorflow
`Optimizers <https://www.tensorflow.org/versions/master/api_docs/python/train.html#Optimizer>`_ in the keyword argument `optimizer_class`.


Learning rate
.............

We have two methods implemented to deal with the update of the learning rate.
The first one is the :py:class:`bob.learn.tensorflow.trainers.constant`, which is just a constant value along the training.
The second one is the :py:class:`bob.learn.tensorflow.trainers.exponential_decay`, which, as the name says, implements
an exponential decay of the learning rate along the training.


Initialization
..............

We have implemented some strategies to initialize the tensorflow variables.
Check it out `Initialization <py_api.html#initialization>`_.


Loss
....

Loss functions must be wrapped as a :py:class:`bob.learn.tensorflow.loss.BaseLoss` objects.
For instance, if you want to use the sparse softmax cross entropy loss between logits and labels you should do like this.

.. code-block:: python

    >>> loss = BaseLoss(tf.nn.sparse_softmax_cross_entropy_with_logits, tf.reduce_mean)

As you can observe, you can pass directly tensorflow operations to this object.

We have also some crafted losses.
For instance, the loss :py:class:`bob.learn.tensorflow.loss.TripletLoss` is used to train triplet networks and the
:py:class:`bob.learn.tensorflow.loss.ContrastiveLoss` is used to train siamese networks.


Analyzers
.........

To be discussed.


Sandbox
-------

We have a sandbox of examples in a git repository `https://gitlab.idiap.ch/tiago.pereira/bob.learn.tensorflow_sandbox`
The sandbox has some example of training:
- MNIST with softmax
- MNIST with Siamese Network
- MNIST with Triplet Network
- Face recognition with MOBIO database
- Face recognition with CASIA WebFace database
