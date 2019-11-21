.. vim: set fileencoding=utf-8 :


===========
 Regression
===========

A flexible estimator for regression problems is implemented in
:py:class:`bob.learn.tensorflow.estimators.Regressor`. You can use this
estimator for various regression problems. The guide below (taken from
https://www.tensorflow.org/tutorials/keras/basic_regression) outlines a basic
regression example using the API of this package.

The Boston Housing Prices dataset
=================================

.. testsetup::

    import tempfile
    model_dir = tempfile.mkdtemp()


1. Let's do some imports:
*************************

.. doctest::

    >>> import tensorflow as tf
    >>> from tensorflow import keras
    >>> import tensorflow.contrib.slim as slim
    >>> from bob.learn.tensorflow.estimators import Regressor

2. Download the dataset:
************************

.. doctest::

    >>> boston_housing = keras.datasets.boston_housing
    >>> print("doctest s**t"); (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data() # doctest: +ELLIPSIS
    doc...
    >>> print("Training set: {}".format(train_data.shape))
    Training set: (404, 13)
    >>> print("Testing set:  {}".format(test_data.shape))
    Testing set:  (102, 13)

3. Normalize features
*********************

.. doctest::

    >>> # Test data is *not* used when calculating the mean and std.
    >>>
    >>> mean = train_data.mean(axis=0)
    >>> std = train_data.std(axis=0)
    >>> train_data = (train_data - mean) / std
    >>> test_data = (test_data - mean) / std

4. Define the input functions
*****************************

.. doctest::

    >>> EPOCH = 2
    >>> def input_fn(mode):
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         features, labels = train_data, train_labels
    ...     else:
    ...         features, labels, = test_data, test_labels
    ...     dataset = tf.data.Dataset.from_tensor_slices((features, labels, [str(x) for x in labels]))
    ...     dataset = dataset.batch(1)
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(len(labels), EPOCH))
    ...     data, label, key = dataset.make_one_shot_iterator().get_next()
    ...     # key is a unique string identifier of each sample.
    ...     # Here we just use the string version of labels.
    ...     return {'data': data, 'key': key}, label
    ...
    >>> def train_input_fn():
    ...     return input_fn(tf.estimator.ModeKeys.TRAIN)
    ...
    >>> def eval_input_fn():
    ...     return input_fn(tf.estimator.ModeKeys.EVAL)


5. Create the estimator
***********************

.. doctest::

    >>> def architecture(data, mode, **kwargs):
    ...     endpoints = {}
    ...
    ...     with tf.variable_scope('DNN'):
    ...
    ...         name = 'fc1'
    ...         net = slim.fully_connected(data, 64, scope=name)
    ...         endpoints[name] = net
    ...
    ...         name = 'fc2'
    ...         net = slim.fully_connected(net, 64, scope=name)
    ...         endpoints[name] = net
    ...
    ...     return net, endpoints
    ...
    >>> estimator = Regressor(architecture, model_dir=model_dir)


5. Train and evaluate the model
*******************************

.. doctest::

    >>> estimator.train(train_input_fn) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE +SKIP
    <bob.learn.tensorflow.estimators.Regressor ...

    >>> 'rmse' in estimator.evaluate(eval_input_fn) # doctest: +SKIP
    True

    >>> list(estimator.predict(eval_input_fn)) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE  +SKIP
    [...

.. testcleanup::

    import shutil
    shutil.rmtree(model_dir, True)
