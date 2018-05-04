.. vim: set fileencoding=utf-8 :


===========
 User guide
===========

This package builds on top of tensorflow_. You are expected to have some
familiarity with it before continuing. We recommend reading at least the
following pages:

* https://www.tensorflow.org/get_started
* https://www.tensorflow.org/programmers_guide
* https://www.tensorflow.org/programmers_guide/estimators
* https://www.tensorflow.org/programmers_guide/datasets

The best way to use tensorflow_ is to use its ``tf.estimator`` and ``tf.data``
API. The estimators are an abstraction API for machine learning models and the
data API is here to help you build complex and efficient input pipelines to
your model.


Face recognition example
========================

Here is a quick code example to build a simple convolutional neural network
(CNN) for recognizing faces from the ATNT database.

1. Let's do some imports:

.. doctest::

    >>> from bob.learn.tensorflow.dataset.bio import BioGenerator
    >>> from bob.learn.tensorflow.utils import to_channels_last
    >>> from bob.learn.tensorflow.estimators import Logits
    >>> import bob.db.atnt
    >>> import tensorflow as tf

2. Define the inputs:

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
    ...     dataset = dataset.cache('/path/to/cache')
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         dataset = dataset.repeat()
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

3. Define the estimator:

.. doctest::

    >>> def architecture(data, mode, **kwargs):
    ...     endpoints = {}
    ...     training = mode == tf.estimator.ModeKeys.TRAIN
    ...
    ...     with tf.variable_scope('CNN'):
    ...
    ...         name = 'conv'
    ...         net = tf.layers.conv2d(data, filters=32, kernel_size=(
    ...             5, 5), strides=2, padding='same', activation=tf.nn.relu, name=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'pool'
    ...         net = tf.layers.max_pooling2d(net, pool_size=(
    ...             2, 2), strides=1, padding='same', name=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'pool-flat'
    ...         net = tf.layers.flatten(net, name=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'dense'
    ...         net = tf.layers.dense(
    ...             net, units=128, activation=tf.nn.relu, name=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'dropout'
    ...         net = tf.layers.dropout(
    ...             inputs=net, rate=0.4, training=training)
    ...         endpoints[name] = net
    ...
    ...     return net, endpoints
    ...
    ...
    >>> estimator = Logits(
    ...     architecture,
    ...     optimizer=tf.train.GradientDescentOptimizer(1e-4),
    ...     loss_op=tf.losses.sparse_softmax_cross_entropy,
    ...     n_classes=20,  # the number of identities in the world set of ATNT database
    ...     embedding_validation=True,
    ...     validation_batch_size=8,
    ... )  # doctest: +SKIP

4. Train and evaluate the model:

.. doctest::

    >>> tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)  # doctest: +SKIP


There are important concepts in this package about how to build input pipelines
and estimators. Let's summarize them.


Data pipeline
-------------

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
to disk for faster reading of data in your training.

The Estimator
-------------

The estimators can also be customized using different architectures, loss
functions, and optimizers.



.. _tensorflow: https://www.tensorflow.org/
