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

from PyMTL.util import logger, pickle_obj

def generate_boolean_function(a, d=8, random_seed=0):
    """Generate a Boolean function in disjunctive normal form according to the
    given parameters.
    
    NOTE: This function implements the Boolean function generation algorithm as
    given in "(2001) Sadohara - Learning of Boolean Functions using Support
    Vector Machines - ALT".
    
    Parameters
    ----------
    a : int
        The number of attributes/variables of the generated function.
    d : int
        The expected number of attributes/variables in a disjunct.
    random_seed : int
        The random seed with which to initialize a private Random object.
    
    Returns
    -------
    attributes : list
        The sympy's Symbol objects representing the attributes of the Boolean
        function.
    function : Boolean function comprised of Boolean operators from sympy.logic
        The Boolean function.
    
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


def generate_examples(attributes, function, n=100, noise=0, random_state=None):
    """Generate examples for the given Boolean function. The values of
    attributes are chosen according to a random uniform distribution. 
    
    Parameters
    ----------
    attributes : list
        The sympy's Symbol objects representing the attributes of the Boolean
        function.
    function : Boolean function comprised of Boolean operators from sympy.logic
        The Boolean function.
    n : int
        The number of example to generate.
    noise: float (in range 0.0 -- 1.0)
        The fraction of examples with the value of the Boolean function chosen
        uniformly at random.
    random_state : int or RandomState
        The random seed or random state to use for pseudo-random number
        generator.
        NOTE: This value is passed to sklearn's check_random_state function
    
    Returns
    -------
    X : numpy.ndarray of shape (n, len(attributes))
        The attribute values of the Boolean function.
    y : numpy.ndarray of shape (n,)
        The values of the Boolean function for each example in X.
    
    """
    random_state = validation.check_random_state(random_state)
    a = len(attributes)
    X = np.zeros((n, a), dtype="int")
    y = np.zeros(n, dtype="int")
    # create a noise mask with indices indicating whether the corresponding
    # example should have its Boolean function value chosen randomly or not
    noise_mask = np.zeros(n, dtype="bool")
    noise_mask[:int(noise * n)] = True
    random_state.shuffle(noise_mask)
    for i in range(n):
        # choose the values of all attributes according to a random uniform
        # distribution
        X_i = list(random_state.random_integers(0, 1, a))
        if noise_mask[i]:
            # choose the value of y at random
            y_i = random_state.random_integers(0, 1)
        else:
            # substitute the attributes in the function with their values
            y_i = function.subs(zip(attributes, X_i))
        X[i] = X_i
        y[i] = y_i
    return X, y


def _generate_boolean_data(a, d, n, g, tg, noise, random_seed,
                           n_learning_sets=1):
    """Generate a synthetic MTL problem of learning Boolean functions according
    to the given parameters.
    
    Parameters
    ----------
    a : int
        Number of attributes/variables of the generated Boolean functions.
    d : int
        The expected number of attributes/variables in a disjunct.
    n : int
        The number of examples for each task to generate.
    g : int
        The number of task groups to generate. Each task group shares the
        same Boolean functions.
    tg : int
        The number of tasks (with their corresponding data) to generate for
        each task group.
    noise : float
        The proportion of examples of each task that have their class values
        determined randomly.
    random_seed : int
        The random seed with which to initialize a private Random object.
    n_learning_sets : int
        The number of different learning sets to create for each task.
    
    Returns
    -------
    tasks : list
        If n_learning_sets == 1, a list of Bunch objects corresponding to
        Boolean function learning tasks.
        Otherwise, a list of lists of Bunch objects, where each list corresponds
        to a set of different learning sets for each task.
    funcs : list
        A list of Boolean functions comprised of Boolean operators from
        sympy.logic, one function for each task group.
    attr : list of sympy's Symbol objects representing the attributes of the
        generated functions
    
    """
    rnd = random.Random(random_seed)
    # generate Boolean functions
    attrs = []
    funcs = []
    for i in range(g):
        attr, func = generate_boolean_function(a, d,
                                               random_seed=rnd.randint(1, 100))
        attrs.append(attr)
        funcs.append(func)
    # generate examples for all tasks 
    tasks = [[] for i in range(g * tg)]
    for i in range(g):
        attr, func = attrs[i], funcs[i]
        attr_names = [str(a_) for a_ in attr]
        for j in range(tg):
            # NOTE: sympy's pretty() function returns a unicode string, so
            # the string literal must also be a unicode string
            descr = (u"Synthetic boolean data for task {} of group {} "
                     "(function: {})".format(j, i, pretty(func,
                                                          wrap_line=False)))
            id = "Group {}, task {}".format(i, j)
            for k in range(n_learning_sets):
                X, y = generate_examples(attr, func, n, noise=noise,
                                         random_state=rnd.randint(1, 100))
                tasks[i * tg + j].append(Bunch(data=X, target=y,
                                               feature_names=attr_names,
                                               DESCR=descr, ID=id))
    if n_learning_sets == 1:
        tasks = [t[0] for t in tasks]
    return tasks, funcs, attr


def _report_about_generated_boolean_mtl_problem(functions, tasks):
    """Log a report about the generated synthetic Boolean MTL problem
    represented by the given functions and tasks.
    Note: The logger object must be a valid Logger.
    
    Parameters
    ----------
    functions : list
        A list of Boolean functions comprised of Boolean operators from
        sympy.logic, one function for each task group.
    tasks : list
        Either a list of Bunch objects corresponding to Boolean function
        learning tasks,
        or a list of lists of Bunch objects, where each list corresponds
        to a set of different learning sets for each task.
    
    """
    logger.debug("Report about the generated synthetic Boolean MTL problem:")
    logger.debug("  Boolean function of each group:")
    for i, func in enumerate(functions):
        # NOTE: sympy's pretty() function returns a unicode string, so the
        # string literal must also be a unicode string
        logger.debug(u"   - Group {}: {}".format(i, pretty(func,
                                                           wrap_line=False)))
    logger.debug("  % of True values in y for each task:")
    sum_true = 0
    sum_total = 0
    for tl in tasks:
        if isinstance(tl, list):
            for i, t in enumerate(tl):
                cur_true = sum(t.target == True)
                cur_len = len(t.target)
                sum_true += cur_true
                sum_total += cur_len
                logger.debug("   - {} (learning set #{}): {}".\
                             format(t.ID, i, cur_true / cur_len))
        else:
            t = tl
            cur_true = sum(t.target == True)
            cur_len = len(t.target)
            sum_true += cur_true
            sum_total += cur_len
            logger.debug("   - {}: {}".format(t.ID, cur_true / cur_len))
    logger.debug("  Average % of True values in y (across all tasks): {}".\
                 format(sum_true / sum_total))


def generate_boolean_data(a, d, n, g, tg, noise, random_seed=1):
    """Generate a synthetic MTL problem of learning Boolean functions according
    to the given parameters.
    Log the report about the generated MTL problem, which includes:
    - the Boolean function of each group,
    - the % of True values in y for each task,
    - the average % of True values in y (across all tasks).
    
    Parameters
    ----------
    a : int
        Number of attributes/variables of the generated Boolean functions.
    d : int
        The expected number of attributes/variables in a disjunct.
    n : int
        The number of examples for each task to generate.
    g : int
        The number of task groups to generate. Each task group shares the
        same Boolean functions.
    tg : int
        The number of tasks (with their corresponding data) to generate for
        each task group.
    noise : float
        The proportion of examples of each task that have their class values
        determined randomly.
    random_seed : int
        The random seed with which to initialize a private Random object.
    
    Returns
    -------
    tasks : list
        A list of Bunch objects corresponding to Boolean function learning
        tasks.
    
    """
    tasks, funcs, _ = _generate_boolean_data(a, d, n, g, tg, noise, random_seed)
    _report_about_generated_boolean_mtl_problem(funcs, tasks)
    return tasks


from itertools import product

def _generate_complete_test_set(attributes, function):
    """Generate the complete testing set for the given Boolean function by
    generating all 2**len(attributes) possible attribute values of the Boolean
    function.
    
    Parameters
    ----------
    attributes : list
        The sympy's Symbol objects representing the attributes of the Boolean
        function.
    function : Boolean function comprised of Boolean operators from sympy.logic
        The Boolean function.
    
    Returns
    -------
    X : numpy.ndarray of shape (2**len(attributes), len(attributes))
        All possible attribute values of the Boolean function.
    y : numpy.ndarray of shape (2**len(attributes),)
        The values of the Boolean function for each example in X.
    
    """
    a = len(attributes)
    X = np.zeros((2**a, a), dtype="int")
    y = np.zeros(2**a, dtype="int")
    # generate all 2**a possible combinations of values of a Boolean attributes 
    for i, X_i in enumerate(product((0, 1), repeat=a)):
        X[i] = X_i
        # substitute the attributes in the function with their values
        y_i = function.subs(zip(attributes, X_i))
        y[i] = y_i
    return X, y


def generate_boolean_data_with_complete_test_sets(a, d, n, g, tg, noise,
        random_seed=1, n_learning_sets=1, funcs_pickle_path=None):
    """Generate a synthetic MTL problem of learning Boolean functions according
    to the given parameters. In addition, create test sets that cover the
    complete attribute space (2**a distinct examples).
    Log the report about the generated MTL problem, which includes:
    - the Boolean function of each group,
    - the % of True values in y for each task,
    - the average % of True values in y (across all tasks).
    
    Parameters
    ----------
    a : int
        Number of attributes/variables of the generated Boolean functions.
    d : int
        The expected number of attributes/variables in a disjunct.
    n : int
        The number of examples for each task to generate.
    g : int
        The number of task groups to generate. Each task group shares the
        same Boolean functions.
    tg : int
        The number of tasks (with their corresponding data) to generate for
        each task group.
    noise : float
        The proportion of examples of each task that have their class values
        determined randomly.
    random_seed : int (optional)
        The random seed with which to initialize a private Random object.
    n_learning_sets : int (optional)
        The number of different learning sets to create for each task.
    funcs_pickle_path : str (optional)
        Path where to pickle the list of generated Boolean functions. 
    
    Returns
    -------
    tasks : list
        If n_learning_sets == 1, a list of Bunch objects corresponding to
        Boolean function learning tasks.
        Otherwise, a list of lists of Bunch objects, where each list corresponds
        to a set of different learning sets for each task.
    tasks_complete_test_sets : list
        A list of (X, y) tuples corresponding to complete testing sets for each
        task.
    
    """
    tasks, funcs, attr = _generate_boolean_data(a, d, n, g, tg, noise,
                            random_seed, n_learning_sets=n_learning_sets)
    if funcs_pickle_path:
        pickle_obj(funcs, funcs_pickle_path)
    
    tasks_complete_test_sets = []
    # generate a complete testing set for each Boolean function
    for func in funcs:
        complete_test_set = _generate_complete_test_set(attr, func)
        # duplicate the generated complete testing set for each task from the
        # current task group 
        for i in range(tg):
            tasks_complete_test_sets.append(complete_test_set)
    
    _report_about_generated_boolean_mtl_problem(funcs, tasks)
    return tasks, tasks_complete_test_sets


if __name__ == "__main__":
    # generate a Boolean function with 8 variables and disjuncts with an average
    # length of 4
    a, d = 8, 4
    attr, func = generate_boolean_function(a, d, random_seed=2)
    # NOTE: sympy's pretty() function returns a unicode string, so the string
    # literal must also be a unicode string
    print u"Boolean function (a={}, d={}): {}".format(a, d,
                                                pretty(func, wrap_line=False))
    X, y = generate_examples(attr, func, n=1000, random_state=10)
    print "% of True values in y: {:.2f}".format(100 * sum(y == True) / len(y))
    X_noise, y_noise = generate_examples(attr, func, n=1000, noise=0.3,
                                         random_state=10)
    
    # try different learning algorithms in scikit-learn and report their
    # cross-validation scores
    from sklearn.linear_model import LogisticRegression
    from sklearn import cross_validation
    lr = LogisticRegression()
    print "Log. reg. scores: ", cross_validation.cross_val_score(lr, X, y, cv=5)
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    print "Dec. tree scores: ", cross_validation.cross_val_score(dt, X, y, cv=5)
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()
    print "Multnomial naive Bayes scores: ", \
        cross_validation.cross_val_score(mnb, X, y, cv=5)
    from sklearn.svm import SVC
    svc_lin = SVC(kernel="poly", coef0=1, degree=5)
    print "SVM (poly.) scores: ", cross_validation.cross_val_score(svc_lin, X,
                                                                   y, cv=5)
    
    # generate a PDF with the learned decision tree model
#    dt = DecisionTreeClassifier()
#    dt.fit(X, y)
#    import StringIO, pydot
#    from sklearn import tree
#    dot_data = StringIO.StringIO() 
#    tree.export_graphviz(dt, out_file=dot_data) 
#    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#    graph.write_pdf("boolean_func_tree.pdf")
    
#    tasks = generate_boolean_data(16, 8, 200, 20, 5, 0.0, random_seed=1)
#    print "Generated a synthetic Boolean MTL problem with {} tasks.".\
#        format(len(tasks))

    tasks, tasks_complete_test_sets = \
        generate_boolean_data_with_complete_test_sets(8, 4, 100, 2, 3, 0.2,
                                                      random_seed=1)
    print "Generated a synthetic Boolean MTL problem with {} tasks.".\
        format(len(tasks))
    
    tasks, tasks_complete_test_sets = \
        generate_boolean_data_with_complete_test_sets(8, 4, 100, 2, 3, 0.2,
            random_seed=1, n_learning_sets=3)
    