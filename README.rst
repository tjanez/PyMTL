.. -*- mode: rst -*-

PyMTL
=====

PyMTL (Python library for Multi-task learning) is a Python module implementing
a Multi-task learning framework built on top of scikit-learn, SciPy and NumPy.
It is licensed under the GNU General Public License version, either version 3
of the License, or (at your option) any later version.

It was started in 2011 by Tadej Janež for the purposes of his PhD thesis on
*discovering clusters of related learning tasks for improving prediction
models of individual tasks*.

It implements an extended version of the *ERM (Error-reduction merging)* multi-task learning method
published by *Tadej Janež et al.* in `International Journal of Advanced Robotic Systems (IJARS) <http://cdn.intechopen.com/pdfs/43887/InTech-Learning_faster_by_discovering_and_exploiting_object_similarities.pdf>`_.

Dependencies
============

PyMTL is tested to work under Python 2.7 (currently, there are no plans to support Python 3).

It has the following dependencies:

 - NumPy >= 1.3
 - SciPy >= 0.7
 - scikit-learn >= 0.14
 - Orange == 2.5 alpha 1
 - Matplotlib >= 0.99.1
 - SymPy >= 0.7.3


Install
=======

Source
------

GIT
~~~

Check out the latest sources with the command::

    git clone git://github.com/tjanez/PyMTL.git

or if you have write privileges::

    git clone git@github.com:tjanez/PyMTL.git

Running
=======

Go to PyMTL's base directory and execute::

  python src/test.py

To change which experiment should run, edit the ``src/test.py`` file and change the value of the ``test_config`` variable.

