#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import pkg_resources
import shutil
import tensorflow as tf

"""
def test_train_script_softmax():
    tf.reset_default_graph()

    directory = "./temp/train-script"
    train_script = pkg_resources.resource_filename(__name__, './data/train_scripts/softmax.py')

    from subprocess import call
    # Start the training
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])

    # Continuing from the last checkpoint
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])
    shutil.rmtree(directory)

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


def test_train_script_triplet():
    tf.reset_default_graph()

    directory = "./temp/train-script"
    train_script = pkg_resources.resource_filename(__name__, './data/train_scripts/triplet.py')

    from subprocess import call
    # Start the training
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])

    # Continuing from the last checkpoint
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])

    shutil.rmtree(directory)

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0


def test_train_script_siamese():
    tf.reset_default_graph()

    directory = "./temp/train-script"
    train_script = pkg_resources.resource_filename(__name__, './data/train_scripts/siamese.py')

    from subprocess import call
    # Start the training
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])

    # Continuing from the last checkpoint
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])

    shutil.rmtree(directory)

    tf.reset_default_graph()
    assert len(tf.global_variables()) == 0
"""
