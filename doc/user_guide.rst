.. vim: set fileencoding=utf-8 :

===========
 User guide
===========

This package builds on top of tensorflow_ (at least 2.3 is needed). You are
expected to have some familiarity with it before continuing. The best way to use
tensorflow_ is to use its ``tf.keras`` and ``tf.data`` API. We recommend reading
at least the following pages:

* https://www.tensorflow.org/tutorials/quickstart/beginner
* https://www.tensorflow.org/tutorials/quickstart/advanced
* https://keras.io/getting_started/intro_to_keras_for_engineers/
* https://keras.io/getting_started/intro_to_keras_for_researchers/
* https://www.tensorflow.org/tutorials/load_data/images
* https://www.tensorflow.org/guide/data

If you were used to Tensorflow 1 API, then reading these pages are also
recommended:

* https://www.tensorflow.org/guide/effective_tf2
* https://www.tensorflow.org/guide/migrate
* https://www.tensorflow.org/guide/upgrade
* https://github.com/tensorflow/community/blob/master/sigs/testing/faq.md

In the rest of this guide, you will learn a few tips and examples on how to:

* Port v1 checkpoints to tf v2 format.
* Create datasets and save TFRecords.
* Create models with custom training and evaluation logic.
* Mixed-precision training
* Multi-GPU and multi-worker training

After reading this page, you may look at a complete example in:
https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/master/examples/MSCeleba_centerloss_mixed_precision_multi_worker.py


Porting V1 Tensorflow checkpoints to V2
=======================================

Take a look at the notebook located at:
https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/master/examples/convert_v1_checkpoints_to_v2.ipynb
for an example.


Creating datasets from data
===========================

If you are working with Bob databases, below is an example of converting them to
``tf.data.Dataset``'s using :any:`bob.learn.tensorflow.data.dataset_using_generator`:

.. testsetup::

    import tempfile
    temp_dir = model_dir = tempfile.mkdtemp()

.. doctest::

    >>> import bob.db.atnt
    >>> from bob.learn.tensorflow.data import dataset_using_generator
    >>> import tensorflow as tf

    >>> db = bob.db.atnt.Database()
    >>> samples = db.objects(groups="world")

    >>> # construct integer labels for each identity in the database
    >>> CLIENT_IDS = (str(f.client_id) for f in samples)
    >>> CLIENT_IDS = list(set(CLIENT_IDS))
    >>> CLIENT_IDS = dict(zip(CLIENT_IDS, range(len(CLIENT_IDS))))

    >>> def reader(sample):
    ...     img = sample.load(db.original_directory, db.original_extension)
    ...     label = CLIENT_IDS[str(sample.client_id)]
    ...     return img, label

    >>> dataset = dataset_using_generator(samples, reader)
    >>> dataset
    <FlatMapDataset shapes: ((112, 92), ()), types: (tf.uint8, tf.int32)>

Create TFRecords from tf.data.Datasets
======================================

Use :any:`bob.learn.tensorflow.data.dataset_to_tfrecord` and
:any:`bob.learn.tensorflow.data.dataset_from_tfrecord` to painlessly convert
**any** ``tf.data.Dataset`` to TFRecords and create datasets back from those
TFRecords:

    >>> from bob.learn.tensorflow.data import dataset_to_tfrecord
    >>> from bob.learn.tensorflow.data import dataset_from_tfrecord
    >>> path = f"{temp_dir}/my_dataset"
    >>> dataset_to_tfrecord(dataset, path)
    >>> dataset = dataset_from_tfrecord(path)
    >>> dataset
    <MapDataset shapes: ((112, 92), ()), types: (tf.uint8, tf.int32)>

There is also a script called ``bob tf dataset-to-tfrecord`` that wraps the
:any:`bob.learn.tensorflow.data.dataset_to_tfrecord` for easy Grid job
submission.

Create models with custom training and evaluation logic
=======================================================

Training models for biometrics recognition (and metric learning in general) is
different from the typical classification problems since the labels during
training and testing are different. We found that overriding the ``compile``,
``train_step``, and ``test_step`` methods as explained in
https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit is the
best trade-off between the control of what happens during training and
evaluation and writing boilerplate code.


Mixed-precision training
========================
When doing mixed precision training: https://www.tensorflow.org/guide/mixed_precision
it is important to scale the loss before computing the gradients.


Multi-GPU and multi-worker training
===================================

It is important that custom metrics and losses do not average their results by the batch
size as the values should be averaged by the global batch size:
https://www.tensorflow.org/tutorials/distribute/custom_training Take a look at custom
metrics and losses in this package for examples of correct implementations.


.. _tensorflow: https://www.tensorflow.org/
