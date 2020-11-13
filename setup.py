#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import dist
from setuptools import setup

from bob.extension.utils import find_packages
from bob.extension.utils import load_requirements

dist.Distribution(dict(setup_requires=["bob.extension"]))


install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(
    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name="bob.learn.tensorflow",
    version=open("version.txt").read().rstrip(),
    description="Bob bindings for tensorflow",
    url="",
    license="BSD",
    author="Tiago de Freitas Pereira",
    author_email="tiago.pereira@idiap.ch",
    keywords="tensorflow",
    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open("README.rst").read(),
    # This line is required for any distutils based packaging.
    include_package_data=True,
    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        # main entry for bob tf cli
        "bob.cli": [
            "tf = bob.learn.tensorflow.scripts.tf:tf",
            "keras = bob.learn.tensorflow.scripts.keras:keras",
        ],
        # bob tf scripts
        "bob.learn.tensorflow.cli": [
            "datasets-to-tfrecords = bob.learn.tensorflow.scripts.datasets_to_tfrecords:datasets_to_tfrecords",
        ],
        # bob keras scripts
        "bob.learn.tensorflow.keras_cli": [
            "fit = bob.learn.tensorflow.scripts.fit:fit",
        ],
    },
    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
