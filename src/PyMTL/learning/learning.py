#
# learning.py
# Contains classes and methods implementing multi-task learning (MTL) methods.
#
# Copyright (C) 2012, 2013 Tadej Janez
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

import logging

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

from PyMTL.sklearn_utils import change_dummy_classes
from PyMTL.util import logger


class MergeAllLearner:
    
    """Learning strategy that merges all tasks, regardless of whether they
    belong to the same behavior class or not.
    
    """
    
    def __call__(self, tasks, base_learner):
        """Run the merging algorithm for the given tasks. Learn a single model
        on the merger of all tasks' data using the given base learner.
        Return a dictionary of data structures computed within this learner.
        It has the following keys:
            task_models -- dictionary mapping from tasks' ids to the learned
                models (in this case, all tasks' ids will map to the same model)
        
        Arguments:
        tasks -- dictionary mapping from tasks' ids to their Task objects
        base_learner -- scikit-learn estimator
        
        """
        # merge learning data of all tasks
        Xs_ys = [t.get_learn_data() for t in tasks.itervalues()]
        Xs, ys = zip(*Xs_ys)
        merged_data = np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
        logger.debug("Merged data has {0[1]} attributes and {0[0]} examples.".\
                     format(merged_data[0].shape))
        # NOTE: The scikit-learn estimator must be cloned to prevent different
        # tasks from having the same classifiers
        model = clone(base_learner)
        model.fit(*merged_data)
        # assign the fitted model to all tasks
        task_models = dict()
        for tid in tasks:
            task_models[tid] = model
        # create and fill the return dictionary
        R = dict()
        R["task_models"] = task_models
        return R

class NoMergingLearner:
    
    """Learning strategy that doesn't merge any tasks. The base learning
    algorithm only uses the data of each task to build its particular model.
    
    """
    
    def __call__(self, tasks, base_learner):
        """Run the merging algorithm for the given tasks. Learn a model using
        the given base learner for each task on its own data (no merging).
        Return a dictionary of data structures computed within this learner.
        It has the following keys:
            task_models -- dictionary mapping from tasks' ids to the learned
                models
        
        Arguments:
        tasks -- dictionary mapping from tasks' ids to their Task objects
        base_learner -- scikit-learn estimator
        
        """
        task_models = dict()
        for tid, task in tasks.iteritems():
            # NOTE: When the number of unique class values is less than 2, we
            # cannot fit an ordinary model (e.g. logistic regression). Instead,
            # we have to use a dummy classifier which is subsequently augmented
            # to handle all the other class values.
            # NOTE: The scikit-learn estimator must be cloned so that each data
            # set gets its own classifier
            learn = task.get_learn_data()
            if len(np.unique(learn[1])) < 2:
                logger.debug("Learning data for task {} has less than 2 class "
                             "values. Using DummyClassifier.".format(tid))
                model = DummyClassifier()
                model.fit(*learn)
                change_dummy_classes(model, np.array([0, 1]))
            else:
                model = clone(base_learner)
                model.fit(*learn)
            task_models[tid] = model
        # create and fill the return dictionary
        R = dict()
        R["task_models"] = task_models
        return R

import random, sys
from collections import Iterable, OrderedDict
from itertools import combinations

from scipy import stats

from PyMTL.learning import testing

def error_reduction(avg_error1, avg_error2, avg_errorM, size1, size2):
    """Compute the error reduction of merging two tasks by comparing the
    weighted average of average prediction errors of models built and tested on
    each task's learning set with the average prediction error of a model
    built and tested on the merger of tasks' learning sets.
    
    Arguments:
    avg_error1 -- float representing the average prediction error of a model
        built and tested on the first task's learning set
    avg_error2 -- float representing the average prediction error of a model
        built and tested on the second task's learning set
    avg_errorM -- float representing the average prediction error of a model
        built and tested on the merger of both tasks' learning sets
    size1 -- integer representing the size of the first task's learning set
    size2 -- integer representing the size of the second task's learning set
    
    """
    return ((size1*avg_error1+size2*avg_error2) / (size1 + size2)) - avg_errorM

def compute_significance(errors1, errors2):
    """Perform a pair-wise one-sided t-test for testing the hypothesis:
    H_0: avg(errors1) >= avg(errors2).
    Return the significance level at which H_0 can be rejected.
    
    Note: This function has been verified to return the same results as the
    following function in R:
    t.test(errors1, errors2, alternative="less", paired=TRUE)
    
    Arguments:
    errors1 -- list of errors of model1
    errors2 -- list of errors of model2
    
    """
    # perform a pair-wise t-test
    t_statistic, p_value = stats.ttest_rel(errors1, errors2)
    # Note: SciPy's stats.ttest_rel function returns the p_value for two-sided
    # t-test; we transform it so it represents the significance level of
    # rejection of the one-sided hypothesis H_0: avg(errors1)-avg(errors2) >= 0
    if t_statistic >= 0:
        p_value = 1 - (p_value/2)
    else:
        p_value = p_value/2
    return p_value

def _convert_id_to_string(m_id):
    """Convert the (merged) task's id to a string by recursively traversing the
    given hierarchical id object.
    
    Arguments:
    m_id -- either a string representing the id of a single task or a
        hierarchically structured tuple of tuples representing the history of
        the merged task
    
    """
    if isinstance(m_id, tuple) and len(m_id) > 1:
        middle = ",".join([_convert_id_to_string(task) for task in m_id])
        return "M("+middle+")"
    else:
        return str(m_id)

def flatten(l):
    """Return a flattened list of the given (arbitrarily) nested iterable of
    iterables (e.g. list of lists).
    
    Arguments:
    l -- (arbitrarily) nested iterable of iterables
    
    """
    flat_l = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, basestring):
            flat_l.extend(flatten(el))
        else:
            flat_l.append(el)
    return flat_l

def convert_merg_history_to_scipy_linkage(merg_history):
    """Convert the given merging history to same format as returned by the
    SciPy's scipy.cluster.hierarchy.linkage function.
    Return a tuple (Z, labels), where:
        Z -- numpy.ndarray of size (len(merg_history), 4) in the format as
            specified in the scipy.cluster.hierarchy.linkage's docstring
        labels -- list of labels representing ids corresponding to each
            consecutive integer
    
    Arguments:
    merg_history -- a list of lists, where each inner list contains ids of
        single and merged tasks (the first in the form of strings and the
        second in the form of tuples) 
    
    """
    # dictionary mapping from task ids to consecutive integers
    id_to_int = OrderedDict()
    # current consecutive integer
    cur_int = 0
    # convert ids of single tasks to consecutive integers
    for merg in merg_history:
        for id in merg:
            if not isinstance(id, tuple):
                id_to_int[id] = cur_int
                cur_int += 1
    # total number of single tasks
    n = len(id_to_int)
    # create a list of labels (i.e. an implicit reverse mapping from consecutive
    # integers to ids)
    labels = [t[0] for t in sorted(id_to_int.iteritems(),
                                   key=lambda t: t[1])]
    # current 'height' and its increment value
    inc = 1
    cur_h = inc
    # convert the merging history to SciPy's linkage format
    Z = np.zeros((n - 1, 4))
    for i, merg in enumerate(merg_history):
        # number of tasks in the current merger
        cur_n = 0
        for j, id in enumerate(merg):
            if id not in id_to_int:
                id_to_int[id] = cur_int
                cur_int += 1
                cur_n += Z[id_to_int[id] - n, 3]
            else:
                cur_n += 1
            Z[i, j] = id_to_int[id]
        Z[i, 3] = cur_n
        # store the current 'height' and increment its value
        Z[i, 2] = cur_h
        cur_h += inc
    return Z, labels

class MergedTask:
    
    """Contains data pertaining to a particular (merged) task and methods for
    extracting this data.
    
    """
    
    def __init__(self, *tasks):
        """Initialize a MergedTask object. Extract the ids and learn data from
        the given Task/MergedTask objects.
        
        Arguments:
        tasks -- list of either Task or MergedTask objects that are to be merged
            into one task
        
        """
        if len(tasks) == 1:
            t = tasks[0]
            self.id = t.id
            # create a copy of both numpy.arrays
            X, y = t.get_learn_data()
            self._learn = X.copy(), y.copy()
        elif len(tasks) == 2:
            # id is a hierarchically structured tuple of tuples representing the
            # history of the merged task (e.g. id "(5, (38, 40))" represents
            # the merged task that initially contained the merger of tasks
            # 38 and 40, which was later merged with task 5)
            self.id = (tasks[0].id, tasks[1].id)
            # combine merging history for tasks that have it already
            self.merg_history = []
            for t in tasks:
                if hasattr(t, "merg_history"):
                    self.merg_history.extend(t.merg_history)
            self.merg_history.append([tasks[0].id, tasks[1].id])
            # merge learning data of all tasks
            Xs_ys = [t.get_learn_data() for t in tasks]
            Xs, ys = zip(*Xs_ys)
            self._learn = np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
        else:
            raise ValueError("Trying to merge more than 2 tasks is not "
                             "possible!")
    
    def __str__(self):
        """Return a "pretty" representation of the merged task by indicating
        which original tasks were merged into this task.
        
        """
        return _convert_id_to_string(self.id)
    
    def get_learn_data(self):
        """Return the learning data of the task."""
        return self._learn
    
    def get_data_size(self):
        """Return the number of instances in task's learning data."""
        return len(self._learn[1])
                
    def get_original_ids(self):
        """Extract original ids of tasks merged into this task.
        Return a list of original task ids.
        
        """
        if isinstance(self.id, tuple):
            return flatten(self.id)
        else:
            # the task has not been merged, return its id in a list
            return [self.id]

def sorted_pair((t1, t2)):
    """Return a lexicographically sorted tuple of the given pair of tasks' ids.
    
    Arguments:
    t1 -- object representing the id of task 1
    t2 -- object representing the id of task 2
    
    """
    return (t1, t2) if t1 <= t2 else (t2, t1)

class CandidatePair():
    
    """Contains data pertaining to a pair of tasks that is a candidate for
    merging.
    
    """
    def __init__(self, t1, t2, p_values):
        """Initialize a CandidatePair object. Compute the appropriate key from
        the given tasks' ids and store the given p_values.
        
        Arguments:
        t1 -- object representing the id of task 1
        t2 -- object representing the id of task 2
        p_values -- dictionary with two keys: "dataM vs data1; dataM" and
            "dataM vs data2; dataM" corresponding to the appropriate p-values
        
        """
        self.key = sorted_pair((t1, t2))
        self.p_values = p_values
    
    def __str__(self):
        """Return a "pretty" representation of the candidate pair of tasks. """ 
        return "({},{})".format(_convert_id_to_string(self.key[0]),
                                _convert_id_to_string(self.key[1]))
    
    def get_max_p_value(self):
        """Return the maximal p-value of the pair of tasks. """
        return max(self.p_values["dataM vs data1; dataM"],
                   self.p_values["dataM vs data2; dataM"])       

def update_progress(progress, width=20, invert=False):
    """Write a textual progress bar to the console along with the progress' 
    numerical value in percent.
    
    Arguments:
    progress -- float in range [0, 1] indicating the progress
    
    Keyword arguments:
    width -- integer representing the width (in characters) of the textual
        progress bar
    invert -- boolean indicating whether the progress' value should be inverted
    
    """
    template = "\r[{:<" + str(width) + "}] {:.1f}%"
    if invert:
        progress = 1 - progress
    sys.stdout.write(template.format('#' * (int(progress * width)),
                                     progress * 100))
    sys.stdout.flush()

class ERMLearner:
    
    """Learning method that intelligently merges data for different tasks that
    exhibit the same or similar behavior. By increasing the number of learning
    examples, the base learning algorithm can build a more accurate model.
    The merging of tasks' data is accomplished by observing the average
    prediction errors of models built on separate and merged tasks' data and
    following a set of criteria that determine whether the merging of data
    would be beneficial or not.
    
    """
    
    def __init__(self, folds, seed, prefilter, error_func=None):
        """Initialize the ERMLearner object. Copy the given arguments to private
        attributes.
        
        Arguments:
        folds -- integer representing the number of folds to use when performing
            cross-validation to estimate errors and significances of merging two
            tasks (in the call of the _estimate_errors_significances() function)
        seed -- integer to be used as a seed for the private Random object
        prefilter -- pre-filter object which can be called with a pair of tasks
            and returns a boolean value indicating whether or not the given pair
            of tasks passes the filtering criteria
        error_func -- function that takes the tuple (y_true, y_pred) as input
            and returns a numpy.array with the error value for each sample (this
            only needs to be specified if the base_learner is a regression
            estimator)
        
        """
        self._folds = folds
        self._random = random.Random(seed)
        self._prefilter = prefilter
        self.error_func = error_func
    
    def __call__(self, tasks, base_learner):
        """Run the merging algorithm for the given tasks. Perform the
        intelligent merging of tasks' data according to the ERM learning method.
        After the merging is complete, build a model for each remaining (merged)
        task and assign this model to each original task of this (merged) task.
        Return a dictionary of data structures computed within this call to ERM.
        It has the following keys:
            task_models -- dictionary mapping from each original task id to its
                model
            dend_info -- list of tuples (one for each merged task) as returned
                by the convert_merg_history_to_scipy_linkage function
        
        Arguments:
        tasks -- dictionary mapping from tasks' ids to their Task objects
        base_learner -- scikit-learn estimator
        
        """
        self._base_learner = base_learner
        # create an ordered dictionary of MergedTask objects from the given
        # dictionary of tasks
        self._tasks = OrderedDict()
        for _, task in sorted(tasks.iteritems()):
            merg_task = MergedTask(task)
            self._tasks[merg_task.id] = merg_task
        # populate the dictionary of task pairs that are candidates for merging
        C = dict()
        pairs = list(combinations(self._tasks, 2))
        n_pairs = len(pairs)
        msg = "Computing candidate pairs for merging ({} pairs)".format(n_pairs)
        logger.debug(msg)
        print msg
        for i, (tid_i, tid_j) in enumerate(pairs):
            if self._prefilter(tid_i, tid_j):
                avg_pred_errs, p_values_ij = \
                    self._estimate_errors_significances(tid_i, tid_j)
                er_ij = error_reduction(avg_pred_errs["data1"]["data1"],
                                        avg_pred_errs["data2"]["data2"],
                                        avg_pred_errs["dataM"]["dataM"],
                                        self._tasks[tid_i].get_data_size(),
                                        self._tasks[tid_j].get_data_size())
                min_ij = min(avg_pred_errs["data1"]["dataM"],
                             avg_pred_errs["data2"]["dataM"])
                if  er_ij >= 0 and avg_pred_errs["dataM"]["dataM"] <= min_ij:
                    cp = CandidatePair(tid_i, tid_j, p_values_ij)
                    C[cp.key] = cp
            update_progress(1.* (i + 1) / n_pairs)
        print
        # iteratively merge the most similar pair of tasks, until such pairs
        # exist
        n_cand = len(C)
        msg = "Processing {} candidate pairs for merging".format(n_cand)
        logger.debug(msg)
        print msg
        while len(C) > 0:
            # find the task pair with the minimal maximal p-value
            maxes = [(cp_key, cp.get_max_p_value()) for cp_key, cp in
                     C.iteritems()]
            (min_tid_i, min_tid_j), _ = min(maxes, key=lambda x: x[1])
            # merge the pair of tasks and update self._tasks
            task_M = MergedTask(self._tasks[min_tid_i], self._tasks[min_tid_j])
            tid_M = task_M.id
            del self._tasks[min_tid_i]
            del self._tasks[min_tid_j]
            self._tasks[tid_M] = task_M
            # remove task pairs that don't exist anymore from C
            for (tid_i, tid_j) in C.keys():
                if ((tid_i == min_tid_i) or (tid_i == min_tid_j) or
                    (tid_j == min_tid_i) or (tid_j == min_tid_j)):
                    del C[(tid_i, tid_j)]
            # find new task pairs that are candidates for merging
            for tid_i in self._tasks:
                if tid_i != tid_M and self._prefilter(tid_i, tid_M):
                    avg_pred_errs, p_values_iM = \
                        self._estimate_errors_significances(tid_i, tid_M)
                    er_iM = error_reduction(avg_pred_errs["data1"]["data1"],
                                            avg_pred_errs["data2"]["data2"],
                                            avg_pred_errs["dataM"]["dataM"],
                                            self._tasks[tid_i].get_data_size(),
                                            self._tasks[tid_M].get_data_size())
                    min_iM = min(avg_pred_errs["data1"]["dataM"],
                                 avg_pred_errs["data2"]["dataM"])
                    if er_iM >= 0 and avg_pred_errs["dataM"]["dataM"] <= min_iM:
                        cp = CandidatePair(tid_i, tid_M, p_values_iM)
                        C[cp.key] = cp
            update_progress(1.* len(C) / n_cand, invert=True)
        print
        # build a model for each remaining (merged) task and store the info
        # for drawing a dendrogram showing the merging history
        task_models = dict()
        dend_info = []
        for merg_task in self._tasks.itervalues():
            # NOTE: When the number of unique class values is less than 2, we
            # cannot fit an ordinary model (e.g. logistic regression). Instead,
            # we have to use a dummy classifier which is subsequently augmented
            # to handle all the other class values.
            # NOTE: The scikit-learn estimator must be cloned so that each
            # (merged) task gets its own classifier
            X, y = merg_task.get_learn_data()
            if len(np.unique(y)) < 2:
                logger.info("Learning data for merged task {} has less than 2 "
                            "class values. Using DummyClassifier.".\
                            format(merg_task))
                model = DummyClassifier()
                model.fit(X, y)
                change_dummy_classes(model, np.array([0, 1]))
            else:
                model = clone(self._base_learner)
                model.fit(X, y)
            # assign this model to each original task of this (merged) task
            original_ids = merg_task.get_original_ids()
            for tid in original_ids:
                task_models[tid] = model
            # store the dendrogram info (if the task is truly a merged task)
            if len(original_ids) > 1:
                dend_info.append(convert_merg_history_to_scipy_linkage(
                                    merg_task.merg_history))
        # create and fill the return dictionary
        R = dict()
        R["task_models"] = task_models
        R["dend_info"] = dend_info
        return R
    
    def _estimate_errors_significances(self, t1, t2):
        """Estimate the average prediction errors of different models on
        selected combinations of the learning sets of tasks t1 and t2.
        Compute the p-values of two one sided t-tests testing the null
        hypotheses:
        - avg_pred_errs["dataM"]["dataM"] >= avg_pred_errs["data1"]["dataM"]
        - avg_pred_errs["dataM"]["dataM"] >= avg_pred_errs["data2"]["dataM"]
        Return a tuple (avg_pred_errs, p_values), where:
            avg_pred_errs -- two-dimensional dictionary with:
                first key corresponding to the name of the learning set,
                second key corresponding to the name of the testing set,
                value corresponding to the average prediction error of the model
                    trained on the learning set and tested on instances from the
                    testing set
            p_values -- dictionary with:
                key corresponding to the tested null hypothesis,
                value corresponding to the p-value of the performed t-test
        
        Arguments:
        t1 -- object representing the id of task 1
        t2 -- object representing the id of task 2
        
        """
        learn1 = self._tasks[t1].get_learn_data()
        learn2 = self._tasks[t2].get_learn_data()
        pred_errs, avg_pred_errs = testing.generalized_cross_validation(
            self._base_learner, learn1, learn2, self._folds,
            self._random.randint(0, 100), self._random.randint(0, 100),
            self.error_func)
        p_values = {}
        # perform a pair-wise one-sided t-test testing H_0:
        # avg_pred_errs["dataM"]["dataM"] >= avg_pred_errs["data1"]["dataM"] 
        p_values["dataM vs data1; dataM"] = compute_significance(
            pred_errs["dataM"]["dataM"], pred_errs["data1"]["dataM"])
        # perform a pair-wise one-sided t-test testing H_0:
        # avg_pred_errs["dataM"]["dataM"] >= avg_pred_errs["data2"]["dataM"]
        p_values["dataM vs data2; dataM"] = compute_significance(
            pred_errs["dataM"]["dataM"], pred_errs["data2"]["dataM"])
        return avg_pred_errs, p_values

if __name__ == "__main__":
    print "TESTING compute_significance()"
    errors1 = [np.random.normal(0.6, 0.2) for i in range(10)]
    errors2 = [np.random.normal(0.8, 0.2) for i in range(10)]
    print "errors1: ", errors1
    print "errors2: ", errors2
    p_value = compute_significance(errors1, errors2)
    print "p-value of of rejection of H_0: avg(errors1) >= avg(errors2): ", \
        p_value
    
    print "TESTING flatten()"
    l = [1, [2, 3, [4, 5, 6], 7], [8, 9]]
    fl = flatten(l)
    print "Original list: ", l
    print "Flattened list: ", fl
