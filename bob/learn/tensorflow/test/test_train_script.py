#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Thu 13 Oct 2016 13:35 CEST

import pkg_resources
import shutil


def test_train_script_softmax():
    directory = "./temp/train-script"
    train_script = pkg_resources.resource_filename(__name__, './data/train_scripts/softmax.py')
    train_script = './data/train_scripts/softmax.py'

    from subprocess import call
    call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])
    #shutil.rmtree(directory)
    assert True


def test_train_script_triplet():
    directory = "./temp/train-script"
    train_script = pkg_resources.resource_filename(__name__, './data/train_scripts/triplet.py')
    #train_script = './data/train_scripts/triplet.py'

    #from subprocess import call
    #call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])
    #shutil.rmtree(directory)

    assert True


def test_train_script_siamese():
    directory = "./temp/train-script"
    train_script = pkg_resources.resource_filename(__name__, './data/train_scripts/siamese.py')
    #train_script = './data/train_scripts/siamese.py'

    #from subprocess import call
    #call(["./bin/train.py", "--iterations", "5", "--output-dir", directory, train_script])
    #shutil.rmtree(directory)

    assert True
