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
from sklearn.utils import validation


def generate_boolean_function(a, d=8):
    """Generate a Boolean function in disjunctive normal form according to the
    given parameters.
    Return a tuple (attributes, function), where:
        attributes -- list of sympy's Symbol objects representing the attributes
            of the generated function
        function -- Boolean function comprised of Boolean operators from
            sympy.logic
    
    Note: This function implements the Boolean function generation algorithm as
    given in "(1997) Domingos, Pazzani - On the Optimality of the Simple
    Bayesian Classifier under Zero-One Loss - ML".
    
    Arguments:
    a -- integer representing the number of attributes/variables of the
        generated function
    
    Keyword arguments:
    d -- integer representing the expected number of attributes/variables in a
        disjunct
    
    """
    attributes = symbols("x1:{}".format(a + 1))
    function = []
    for i in range(2**d - 1):
        disjunct = []
        for attr in attributes:
            if random.random() < d / a:
                if random.random() < 0.5:
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


if __name__ == "__main__":
    random.seed(1)
    attr, func = generate_boolean_function(4, 2)
    X, y = generate_examples(attr, func, n=10, random_state=10)
    print func
    print X, y
    