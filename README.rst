.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. Thu 30 Jan 08:46:53 2014 CET


.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.learn.tensorflow/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bob/bob.learn.tensorflow/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.learn.tensorflow/badges/master/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.learn.tensorflow/commits/master
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.learn.tensorflow
.. image:: http://img.shields.io/pypi/v/bob.learn.tensorflow.png
   :target: https://pypi.python.org/pypi/bob.learn.tensorflow

===========================
 Bob support for tensorflow
===========================

This package is part of the signal-processing and machine learning toolbox
Bob_. It provides tools to run comparable and reproducible biometric
recognition experiments on publicly available databases.

The `User Guide`_ provides installation and usage instructions.

Installation
------------

Follow our `installation`_ instructions. Then, using the Python interpreter
provided by the distribution, bootstrap and buildout this package::

  $ python bootstrap-buildout.py
  $ ./bin/buildout


.. warning:: We assume that `tensorflow`_ is already installed.

Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://gitlab.idiap.ch/bob/bob/wikis/Installation
.. _mailing list: https://groups.google.com/forum/?fromgroups#!forum/bob-devel
.. _user guide: http://pythonhosted.org/bob.bio.base
.. _tensorflow: https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html
