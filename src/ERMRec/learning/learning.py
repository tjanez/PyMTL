#
# learning.py
# Contains classes and methods implementing the merging learning methods.
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

# create a child logger of the ERMRec logger
logger = logging.getLogger("ERMRec.learning.learning")

class MergeAllLearner:
    
    """Learning strategy that merges all users, regardless of whether they
    belong to the same behavior class or not.
    
    """
    
    def __call__(self, users, base_learner):
        """Run the merging algorithm for the given users. Learn a single model
        on the merger of all users' data using the given base learner.
        Return a dictionary mapping from users' ids to the learned models (in
        this case, all users' ids will map to the same model).
        
        Arguments:
        users -- dictionary mapping from users' ids to their User objects
        base_learner -- scikit-learn estimator
        
        """
        # merge learning data of all users
        Xs_ys = [u.get_learn_data() for u in users.itervalues()]
        Xs, ys = zip(*Xs_ys)
        merged_data = np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
        logger.debug("Merged data has {0[1]} attributes and {0[0]} examples.".\
                     format(merged_data[0].shape))
        # NOTE: The scikit-learn estimator must be cloned to prevent different
        # users from having the same classifiers
        base_learner = clone(base_learner)
        base_learner.fit(*merged_data)
        # assign the fitted classifier to all users
        user_models = dict()
        for user_id in users:
            user_models[user_id] = base_learner
        return user_models

class NoMergingLearner:
    
    """Learning strategy that doesn't merge any users. The base learning
    algorithm only uses the data of each user to build its particular model.
    
    """
    
    def __call__(self, users, base_learner):
        """Run the merging algorithm for the given users. Learn a model using
        the given base learner for each user on its own data (no merging).
        Return a dictionary mapping from users' ids to the learned models.
        
        Arguments:
        users -- dictionary mapping from users' ids to their User objects
        base_learner -- scikit-learn estimator
        
        """
        user_models = dict()
        for user_id, user in users.iteritems():
            # NOTE: The scikit-learn estimator must be cloned so that each user
            # gets its own classifier
            base_learner = clone(base_learner)
            user_models[user_id] = base_learner.fit(*user.get_learn_data())
        return user_models

import random
from collections import Iterable, OrderedDict
from itertools import combinations

from scipy import stats

from ERMRec.learning import testing

def error_reduction(avg_error1, avg_error2, avg_errorM, size1, size2):
    """Compute the error reduction of merging two objects by comparing the
    weighted average of average prediction errors of modules built and tested on
    each object's learning set with the average prediction error of a model
    built and tested on the merger of objects' learning sets.
    
    Arguments:
    avg_error1 -- float representing the average prediction error of a model
        built and tested on the first object's learning set
    avg_error2 -- float representing the average prediction error of a model
        built and tested on the second object's learning set
    avg_errorM -- float representing the average prediction error of a model
        built and tested on the merger of both objects' learning sets
    size1 -- integer representing the size of the first object's learning set
    size2 -- integer representing the size of the second object's learning set
    
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
    """Convert the (merged) user's id to a string by recursively traversing the
    given hierarchical id object.
    
    Arguments:
    m_id -- either a string representing the id of a single user or a
        hierarchically structured tuple of tuples representing the history of
        the merged user
    
    """
    if isinstance(m_id, tuple) and len(m_id) > 1:
        middle = ",".join([_convert_id_to_string(user) for user in m_id])
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

class MergedUser:
    
    """Contains data pertaining to a particular (merged) user and methods for
    extracting this data.
    
    """
    
    def __init__(self, *users):
        """Initialize a MergedUser object. Extract the ids and learn data from
        the given User/MergedUser objects.
        
        Arguments:
        users -- list of either User or MergedUser objects that are to be merged
            into one user
        
        """
        if len(users) == 1:
            u = users[0]
            self.id = u.id
            # create a copy of both numpy.arrays
            X, y = u.get_learn_data()
            self._learn = X.copy(), y.copy()
        else:
            # id is a hierarchically structured tuple of tuples representing the
            # history of the merged user (e.g. id "(5, (38, 40))" represents
            # the merged user that initially contained the merger of users
            # 38 and 40, which was later merged with user 5)
            self.id = tuple([u.id for u in users])
            # merge learning data of all users
            Xs_ys = [u.get_learn_data() for u in users]
            Xs, ys = zip(*Xs_ys)
            self._learn = np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
    
    def __str__(self):
        """Return a "pretty" representation of the merged user by indicating
        which original users were merged into this user.
        
        """
        return _convert_id_to_string(self.id)
    
    def get_learn_data(self):
        """Return the learning data of the user."""
        return self._learn
    
    def get_data_size(self):
        """Return the number of instances in user's learning data."""
        return len(self._learn[1])
                
    def get_original_ids(self):
        """Extract original ids of users merged into this user.
        Return a list of original user ids.
        
        """
        if isinstance(self.id, tuple):
            return flatten(self.id)
        else:
            # the user has not been merged, return its id in a list
            return [self.id]

def sorted_pair((u_1, u_2)):
    """Return a lexicographically sorted tuple of the given pair of users' ids.
    
    Arguments:
    u_1 -- object representing the id of user 1
    u_2 -- object representing the id of user 2
    
    """
    return (u_1, u_2) if u_1 <= u_2 else (u_2, u_1)

class CandidatePair():
    
    """Contains data pertaining to a pair of users that is a candidate for
    merging.
    
    """
    def __init__(self, u_1, u_2, p_values):
        """Initialize a CandidatePair object. Compute the appropriate key from
        the given users' ids and store the given p_values.
        
        Arguments:
        u_1 -- object representing the id of user 1
        u_2 -- object representing the id of user 2
        p_values -- dictionary with two keys: "dataM vs data1; dataM" and
            "dataM vs data2; dataM" corresponding to the appropriate p-values
        
        """
        self.key = sorted_pair((u_1, u_2))
        self.p_values = p_values
    
    def __str__(self):
        """Return a "pretty" representation of the candidate pair of users. """ 
        return "({},{})".format(_convert_id_to_string(self.key[0]),
                                _convert_id_to_string(self.key[1]))
    
    def get_max_p_value(self):
        """Return the maximal p-value of the pair of users. """
        return max(self.p_values["dataM vs data1; dataM"],
                   self.p_values["dataM vs data2; dataM"])       

class ERMLearner:
    
    """Learning method that intelligently merges data for different users that
    exhibit the same or similar behavior. By increasing the number of learning
    examples, the base learning algorithm can build a more accurate model.
    The merging of users' data is accomplished by observing the average
    prediction errors of models built on separate and merged objects' data and
    following a set of criteria that determine whether the merging of data
    would be beneficial or not.
    
    """
    
    def __init__(self, folds, seed, prefilter):
        """Initialize the ERMLearner object. Copy the given arguments to private
        attributes.
        
        Arguments:
        folds -- integer representing the number of folds to use when performing
            cross-validation to estimate errors and significances of merging two
            users (in the call of the _estimate_errors_significances() function)
        seed -- integer to be used as a seed for the private Random object
        prefilter -- pre-filter object which can be called with a pair of users
            and returns a boolean value indicating whether or not the given pair
            of users passes the filtering criteria
        
        """
        self._folds = folds
        self._random = random.Random(seed)
        self._prefilter = prefilter
    
    def __call__(self, users, base_learner):
        """Run the merging algorithm for the given users. Perform the
        intelligent merging of users' data according to the ERM learning method.
        After the merging is complete, build a model for each remaining (merged)
        user and assign this model to each original user of this (merged) user.
        
        Arguments:
        users -- dictionary mapping from users' ids to their User objects
        base_learner -- scikit-learn estimator
        
        """
        self._base_learner = base_learner
        # create an ordered dictionary of MergedUser objects from the given
        # dictionary of users
        self._users = OrderedDict()
        for _, u_obj in sorted(users.iteritems()):
            merg_u_obj = MergedUser(u_obj)
            self._users[merg_u_obj.id] = merg_u_obj
        # populate the dictionary of user pairs that are candidates for merging
        C = dict()
        for u_i, u_j in combinations(self._users, 2):
            if self._prefilter(u_i, u_j):
                avg_pred_errs, p_values_ij = \
                    self._estimate_errors_significances(u_i, u_j)
                er_ij = error_reduction(avg_pred_errs["data1"]["data1"],
                                        avg_pred_errs["data2"]["data2"],
                                        avg_pred_errs["dataM"]["dataM"],
                                        self._users[u_i].get_data_size(),
                                        self._users[u_j].get_data_size())
                min_ij = min(avg_pred_errs["data1"]["dataM"],
                             avg_pred_errs["data2"]["dataM"])
                if  er_ij >= 0 and avg_pred_errs["dataM"]["dataM"] <= min_ij:
                    cp = CandidatePair(u_i, u_j, p_values_ij)
                    C[cp.key] = cp
        # iteratively merge the most similar pair of users, until such pairs
        # exist
        while len(C) > 0:
            # find the object pair with the minimal maximal p-value
            maxes = [(cp_key, cp.get_max_p_value()) for cp_key, cp in
                     C.iteritems()]
            (min_u_i, min_u_j), _ = min(maxes, key=lambda x: x[1])
            # merge the pair of users and update self._users
            u_M_obj = MergedUser(self._users[min_u_i], self._users[min_u_j])
            u_M = u_M_obj.id
            del self._users[min_u_i]
            del self._users[min_u_j]
            self._users[u_M] = u_M_obj
            # remove object pairs that don't exist anymore from C
            for (u_i, u_j) in C.keys():
                if ((u_i == min_u_i) or (u_i == min_u_j) or
                    (u_j == min_u_i) or (u_j == min_u_j)):
                    del C[(u_i, u_j)]
            # find new user pairs that are candidates for merging
            for u_i in self._users:
                if u_i != u_M and self._prefilter(u_i, u_M):
                    avg_pred_errs, p_values_iM = \
                        self._estimate_errors_significances(u_i, u_M)
                    er_iM = error_reduction(avg_pred_errs["data1"]["data1"],
                                            avg_pred_errs["data2"]["data2"],
                                            avg_pred_errs["dataM"]["dataM"],
                                            self._users[u_i].get_data_size(),
                                            self._users[u_M].get_data_size())
                    min_iM = min(avg_pred_errs["data1"]["dataM"],
                                 avg_pred_errs["data2"]["dataM"])
                    if er_iM >= 0 and avg_pred_errs["dataM"]["dataM"] <= min_iM:
                        cp = CandidatePair(u_i, u_M, p_values_iM)
                        C[cp.key] = cp
        # build a model for each remaining (merged) user
        user_models = dict()
        for merg_u_obj in self._users.itervalues():
            # NOTE: The scikit-learn estimator must be cloned so that each
            # (merged) user gets its own classifier
            base_learner = clone(self._base_learner)
            base_learner.fit(*merg_u_obj.get_learn_data())
            # assign this model to each original user of this (merged) user
            for user_id in merg_u_obj.get_original_ids():
                user_models[user_id] = base_learner
        return user_models
    
    def _estimate_errors_significances(self, u_1, u_2):
        """Estimate the average prediction errors of different models on
        selected combinations of the learning sets of users u_1 and u_2.
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
        u_1 -- object representing the id of user 1
        u_2 -- object representing the id of user 2
        
        """
        learn_1 = self._users[u_1].get_learn_data()
        learn_2 = self._users[u_2].get_learn_data()
        pred_errs, avg_pred_errs = testing.generalized_cross_validation(
            self._base_learner, learn_1, learn_2, self._folds,
            self._random.randint(0, 100), self._random.randint(0, 100))
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
#    # TEST compute_significance()
#    errors1 = [np.random.normal(0.6, 0.2) for i in range(10)]
#    errors2 = [np.random.normal(0.8, 0.2) for i in range(10)]
#    print "errors1: ", errors1
#    print "errors2: ", errors2
#    p_value = compute_significance(errors1, errors2)
#    print "p-value of of rejection of H_0: avg(errors1) >= avg(errors2): ", p_value
    
    # TEST flatten()
    l = [1, [2, 3, [4, 5, 6], 7], [8, 9]]
    fl = flatten(l)
    print "Original list: ", l
    print "Flattened list: ", fl
    