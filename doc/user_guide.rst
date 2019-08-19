.. vim: set fileencoding=utf-8 :


===========
 User guide
===========

This package builds on top of tensorflow_. You are expected to have some
familiarity with it before continuing. We recommend reading at least the
following pages:

* https://www.tensorflow.org/get_started
* https://www.tensorflow.org/guide/
* https://www.tensorflow.org/guide/estimators
* https://www.tensorflow.org/guide/datasets

The best way to use tensorflow_ is to use its ``tf.estimator`` and ``tf.data``
API. The estimators are an abstraction API for machine learning models and the
data API is here to help you build complex and efficient input pipelines to
your model. Using the estimators and dataset API of tensorflow will make your
code more complex but instead you will enjoy more efficiency and avoid code
redundancy.


Face recognition example using bob.db databases
===============================================


Let's take a look at a complete example of using a convolutional neural network
(CNN) for recognizing faces from the ATNT database. At the end, we will explain
the data pipeline in more detail.

1. Let's do some imports:
*************************

.. testsetup::

    import tempfile
    temp_dir = model_dir = tempfile.mkdtemp()

.. doctest::

    >>> from bob.learn.tensorflow.dataset.bio import BioGenerator
    >>> from bob.learn.tensorflow.utils import to_channels_last
    >>> from bob.learn.tensorflow.estimators import Logits
    >>> import bob.db.atnt
    >>> import tensorflow as tf
    >>> import tensorflow.contrib.slim as slim

2. Define the inputs:
*********************

.. _input_fn:

.. doctest::

    >>> def input_fn(mode):
    ...     db = bob.db.atnt.Database()
    ...
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         groups = 'world'
    ...     elif mode == tf.estimator.ModeKeys.EVAL:
    ...         groups = 'dev'
    ...
    ...     files = db.objects(groups=groups)
    ...
    ...     # construct integer labels for each identity in the database
    ...     CLIENT_IDS = (str(f.client_id) for f in files)
    ...     CLIENT_IDS = list(set(CLIENT_IDS))
    ...     CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))
    ...
    ...     def biofile_to_label(f):
    ...         return CLIENT_IDS[str(f.client_id)]
    ...
    ...     def load_data(database, f):
    ...         img = f.load(database.original_directory, database.original_extension)
    ...         # make a channels_first image (bob format) with 1 channel
    ...         img = img.reshape(1, 112, 92)
    ...         return img
    ...
    ...     generator = BioGenerator(db, files, load_data, biofile_to_label)
    ...
    ...     dataset = tf.data.Dataset.from_generator(
    ...         generator, generator.output_types, generator.output_shapes)
    ...
    ...     def transform(image, label, key):
    ...         # convert to channels last
    ...         image = to_channels_last(image)
    ...
    ...         # per_image_standardization
    ...         image = tf.image.per_image_standardization(image)
    ...         return (image, label, key)
    ...
    ...     dataset = dataset.map(transform)
    ...     dataset = dataset.cache(temp_dir)
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         dataset = dataset.repeat(1)
    ...     dataset = dataset.batch(8)
    ...
    ...     data, label, key = dataset.make_one_shot_iterator().get_next()
    ...     return {'data': data, 'key': key}, label
    ...
    ...
    >>> def train_input_fn():
    ...     return input_fn(tf.estimator.ModeKeys.TRAIN)
    ...
    ...
    >>> def eval_input_fn():
    ...     return input_fn(tf.estimator.ModeKeys.EVAL)
    ...
    ...
    >>> # supply this hook for debugging
    >>> # from tensorflow.python import debug as tf_debug
    >>> # hooks = [tf_debug.LocalCLIDebugHook()]
    >>> hooks = None
    ...
    >>> train_spec = tf.estimator.TrainSpec(
    ...     input_fn=train_input_fn, max_steps=50, hooks=hooks)
    >>> eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

3. Define the architecture:
***************************

.. doctest::

    >>> def architecture(data, mode, **kwargs):
    ...     endpoints = {}
    ...     training = mode == tf.estimator.ModeKeys.TRAIN
    ...
    ...     with tf.variable_scope('CNN'):
    ...
    ...         name = 'conv'
    ...         net = slim.conv2d(data, 32, kernel_size=(
    ...             5, 5), stride=2, padding='SAME', activation_fn=tf.nn.relu, scope=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'pool'
    ...         net = slim.max_pool2d(net, (2, 2),
    ...             stride=1, padding='SAME', scope=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'pool-flat'
    ...         net = slim.flatten(net, scope=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'dense'
    ...         net = slim.fully_connected(net, 128, scope=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'dropout'
    ...         net = slim.dropout(
    ...             inputs=net, keep_prob=0.4, is_training=training)
    ...         endpoints[name] = net
    ...
    ...     return net, endpoints


.. important ::

    Practical advice: use ``tf.contrib.slim`` to craft your CNNs. Although
    Tensorflow's documentation recommend the usage of ``tf.layers`` and
    ``tf.keras``, in our experience ``slim`` has better defaults and is more
    integrated with tensorflow's framework (compared to ``tf.keras``),
    probably because it is used more often internally at Google.


4. Estimator:
************************

Explicitly triggering the estimator
...................................

.. doctest::

    >>> estimator = Logits(
    ...     architecture,
    ...     optimizer=tf.train.GradientDescentOptimizer(1e-4),
    ...     loss_op=tf.losses.sparse_softmax_cross_entropy,
    ...     n_classes=20,  # the number of identities in the world set of ATNT database
    ...     embedding_validation=True,
    ...     validation_batch_size=8,
    ...     model_dir=model_dir,
    ... )
    >>> tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec) # doctest: +SKIP
    ({'accuracy':...


Triggering the estimator via command line
..........................................

In the example above we explicitly triggered the training and validation via
`tf.estimator.train`. We provide command line scripts that does that for you.

Check the command bellow fro training::

 $ bob tf train --help

and to evaluate::

 $ bob tf eval --help


Data pipeline
=============

There are several ways to provide data to Tensorflow graphs. In this section we
provide some examples on how to make the bridge between `bob.db` databases and
tensorflow `input_fn`.

The BioGenerator input pipeline
*******************************

The :any:`bob.learn.tensorflow.dataset.bio.BioGenerator` class can be used to
convert any database of bob (not just bob.bio.base's databases) to a
``tf.data.Dataset`` instance.

While building the input pipeline, you can manipulate your data in two
sections:

* In the ``load_data`` function where everything is a numpy array.
* In the ``transform`` function where the data are tensorflow tensors.

For example, you can annotate, crop to bounding box, and scale your images in
the ``load_data`` function and apply transformations on images (e.g. random
crop, mean normalization, random flip, ...) in the ``transform`` function.

Once these transformations are applied on your data, you can easily cache them
to disk (using ``tf.data.Dataset.cache``) for faster reading of data in your
training.


Input pipeline with TFRecords
*****************************

An optimized way to provide data to Tensorflow graphs is using tfrecords. In
this `link <http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/>`_
you have a very nice guide on how TFRecord works.

In `bob.learn.tensorflow` we provide a command line interface
``bob tf db_to_tfrecords`` that converts ``bob.db`` databases to TFRecords.
Type the snippet bellow for help::

  $ bob tf db_to_tfrecords --help


To generate a tfrecord for our
`Face recognition example using bob.db databases`_ example use the following
snippet.

.. doctest::

    >>> from bob.bio.base.utils import read_original_data
    >>> from bob.bio.base.test.dummy.database import database # this is based on bob.db.atnt

    >>> groups = 'dev'

    >>> samples = database.all_files(groups=groups)

    >>> CLIENT_IDS = (str(f.client_id) for f in database.objects(groups=groups))
    >>> CLIENT_IDS = set(CLIENT_IDS)
    >>> CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))

    >>> def file_to_label(f):
    ...     return CLIENT_IDS[str(f.client_id)]

    >>> def reader(biofile):
    ...     data = read_original_data(biofile, database.original_directory, database.original_extension)
    ...     label = file_to_label(biofile)
    ...     key = biofile.path
    ...     return (data, label, key)


After saving this snippet in a python file (let's say `tfrec.py`) run the
following command ::

    $ bob tf db_to_tfrecords tfrec.py -o atnt.tfrecord

Once this is done you can replace the `input_fn`_ defined above by the snippet
bellow.

.. doctest::

    >>>
    >>> from bob.learn.tensorflow.dataset.tfrecords import shuffle_data_and_labels_image_augmentation
    >>>
    >>> tfrecords_filename = ['/path/to/atnt.tfrecord']
    >>> data_shape = (112, 92 , 3)
    >>> data_type = tf.uint8
    >>> batch_size = 16
    >>> epochs = 1
    >>>
    >>> def train_input_fn():
    ...     return shuffle_data_and_labels_image_augmentation(
    ...                tfrecords_filename,
    ...                data_shape,
    ...                data_type,
    ...                batch_size,
    ...                epochs=epochs)

.. testcleanup::

    import shutil
    shutil.rmtree(model_dir, True)

The Estimator
=============

In this package we have crafted 4 types of estimators.

   - Logits: `Cross entropy loss
     <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_
     in the hot-encoded layer
     :py:class:`bob.learn.tensorflow.estimators.Logits`
   - LogitsCenterLoss: `Cross entropy loss
     <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_
     PLUS the `center loss <https://ydwen.github.io/papers/WenECCV16.pdf>`_ in
     the hot-encoded layer
     :py:class:`bob.learn.tensorflow.estimators.LogitsCenterLoss`
   - Siamese: Siamese network estimator
     :py:class:`bob.learn.tensorflow.estimators.Siamese`
   - Triplet: Triplet network estimator
     :py:class:`bob.learn.tensorflow.estimators.Triplet`

.. _tensorflow: https://www.tensorflow.org/

