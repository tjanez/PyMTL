#
# synthetic_data.py
# Contains methods for generating synthetic multi-task learning (MTL) data sets.
#
# Copyright (C) 2013 Tadej Janez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author(s): Tadej Janez <tadej.janez@fri.uni-lj.si>
#

from __future__ import division

import random

import numpy as np
from sympy import symbols
from sympy.logic import And, Or, Not
from sympy.printing import pretty
from sklearn.utils import validation
from sklearn.datasets.base import Bunch

from PyMTL.util import logger

def generate_boolean_function(a, d=8, random_seed=0):
    """Generate a Boolean function in disjunctive normal form according to the
    given parameters.
    Return a tuple (attributes, function), where:
        attributes -- list of sympy's Symbol objects representing the attributes
            of the generated function
        function -- Boolean function comprised of Boolean operators from
            sympy.logic
    
    Note: This function implements the Boolean function generation algorithm as
    given in "(2001) Sadohara - Learning of Boolean Functions using Support
    Vector Machines - ALT".
    
    Arguments:
    a -- integer representing the number of attributes/variables of the
        generated function
    
    Keyword arguments:
    d -- integer representing the expected number of attributes/variables in a
        disjunct
    random_seed -- integer representing the random seed with which to initialize
        a private Random object
    
    """
    rand_obj = random.Random(random_seed)
    attributes = symbols("x1:{}".format(a + 1))
    function = []
    for i in range(2 ** (d - 2)):
        disjunct = []
        for attr in attributes:
            if rand_obj.random() < d / a:
                if rand_obj.random() < 0.5:
                    disjunct.append(attr)
                else:
                    disjunct.append(Not(attr))
        disjunct = And(*disjunct)
        function.append(disjunct)
    return attributes, Or(*function)


def generate_examples(attributes, function, n=100, random_state=None):
    """Generate examples for the given Boolean function. The values of
    attributes are chosen according to a random uniform distribution. 
    Return a tuple (X, y), where:
        X -- numpy.ndarray of shape (n, len(attributes)) representing the values
            of attributes
        y -- numpy.ndarray of shape (n,) representing the values of the Boolean
            function for each example in X
    
    Arguments:
    attributes -- list of sympy's Symbol objects representing the attributes
        of the Boolean function
    function -- Boolean function comprised of Boolean operators from sympy.logic
    
    Keyword arguments:
    n -- integer representing the number of examples to generate
    random_state -- integer or RandomState representing the state used for
        pseudo-random number generator; this value is passed to sklearn's
        check_random_state function
    
    """
    random_state = validation.check_random_state(random_state)
    a = len(attributes)
    X = np.zeros((n, a), dtype="bool")
    y = np.zeros(n, dtype="bool")
    for i in range(n):
        # choose the values of all attributes according to a random uniform
        # distribution
        X_i = list(random_state.random_integers(0, 1, a))
        # substitute the attributes in the function with their values
        y_i = function.subs(zip(attributes, X_i))
        X[i] = X_i
        y[i] = y_i
    return X, y


def generate_boolean_data(a, d, n, g, tg, noise, random_seed=1):
    """Generate a synthetic MTL problem of learning Boolean functions according
    to the given parameters.
    Log the % of True values in y for each task.
    
    Parameters
    ----------
    a -- int
        Number of attributes/variables of the generated Boolean functions.
    d -- int
        The expected number of attributes/variables in a disjunct.
    n -- int
        The number of examples for each task to generate.
    g -- int
        The number of task groups to generate. Each task group shares the
        same Boolean functions.
    tg -- int
        The number of tasks (with their corresponding data) to generate for
        each task group.
    noise -- float
        The proportion of examples of each task that have their class values
        determined randomly.
    random_seed -- int
        The random seed with which to initialize a private Random object.
    
    Returns
    -------
    tasks -- list
        A list of Bunch objects that correspond to regression tasks, each task
        corresponding to one subject.
    
    """
    rnd = random.Random(random_seed)
    tasks = []
    for i in range(g):
        attr, func = generate_boolean_function(a, d,
                                               random_seed=rnd.randint(1, 100))
        for j in range(tg):
            X, y = generate_examples(attr, func, n,
                                     random_state=rnd.randint(1, 100))
            descr = "Synthetic boolean data for task {} of group {} " \
                "(function: {})".format(j, i, pretty(func))
            id = "Task {} of group {}".format(j, i)
            tasks.append(Bunch(data=X,
                               target=y,
                               DESCR=descr,
                               ID=id))
    logger.debug("Report about the generated synthetic Boolean MTL problem:")
    logger.debug("% of True values in y for each task:")
    sum_true = 0
    sum_total = 0
    for t in tasks:
        cur_true = sum(t.target == True)
        cur_len = len(t.target)
        sum_true += cur_true
        sum_total += cur_len
        logger.debug("{}: {}".format(t.ID, cur_true / cur_len))
    logger.debug("Average % of True values in y (across all tasks): {}".\
                 format(sum_true / sum_total))
    return tasks


if __name__ == "__main__":
    a, d = 4, 2
    attr, func = generate_boolean_function(a, d, random_seed=1)
    print "Boolean function (a={}, d={}):".format(a, d), pretty(func)
    X, y = generate_examples(attr, func, n=10000, random_state=10)
    print "% of True values in y: {:.2f}".format(100 * sum(y == True) / len(y))

    tasks = generate_boolean_data(16, 8, 200, 20, 5, 0.0, random_seed=1)
    print "Generated a synthetic Boolean MTL problem with {} tasks.".\
        format(len(tasks))
