#
# testing.py
# Contains classes and methods for internal testing used by ERM MTL method.
#
# Copyright (C) 2011, 2012, 2013 Tadej Janez
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
from sklearn import cross_validation, pipeline
from sklearn.dummy import DummyClassifier

from PyMTL.sklearn_utils import change_dummy_classes

# create a child logger of the PyMTL logger
logger = logging.getLogger("PyMTL.learning.testing")

def _compute_average_prediction_errors(pred_errs):
    """Compute average prediction errors from the given prediction error lists
    (or numpy.arrays).
    Return a two-dimensional dictionary with:
        first key corresponding to the name of the learning set,
        second key corresponding to the name of the testing set,
        value corresponding to the average prediction error of the model
            trained on the learning set and tested on instances from the
            testing set
    
    Arguments:
    pred_errs -- two-dimensional dictionary with:
        first key corresponding to the name of the learning set,
        second key corresponding to the name of the testing set,
        value corresponding to the list (or numpy.array) of prediction errors
            using a model trained on the learning set and tested on instances
            from the testing set
    
    """
    avg_pred_errs = {}
    for learn_data in pred_errs:
        avg_pred_errs[learn_data] = {}
        for test_data in pred_errs[learn_data]:
            avg_pred_errs[learn_data][test_data] = \
                np.mean(pred_errs[learn_data][test_data])
    return avg_pred_errs

def _check_classes(estimator):
    """Check that the list of classes of the given estimator is an ordered list
    of integers ranging from 0 to len(classes) - 1.
    Raise a ValueError if not.
    
    Arguments:
    estimator -- scikit-learn estimator or scikit-learn Pipeline where the
        last step is an estimator
    
    """
    if hasattr(estimator, "classes_"):
        classes = estimator.classes_
    elif (isinstance(estimator, pipeline.Pipeline) and
          hasattr(estimator.steps[-1][1], "classes_")):
        classes = estimator.steps[-1][1].classes_
    else:
        raise ValueError("The given estimator does not have a 'classes_' "
                         "property")
    for i in range(len(classes)):
        if classes[i] != i:
            raise ValueError("The estimator's list of classes is not an ordered"
                             "list of integers.")

def generalized_leave_one_out(learner, data1, data2):
    """Perform a generalized version of the leave-one-out cross-validation
    testing method on the given data sets.
    Estimate the prediction errors of models built on all combinations of the
    data sets (data1, data2 and merged data set) and tested across all
    combinations of the data sets (data1, data2 and merged data set).
    Return a tuple (pred_errs, avg_pred_errs), where:
        pred_errs -- two-dimensional dictionary with:
            first key corresponding to the name of the learning set,
            second key corresponding to the name of the testing set,
            value corresponding to the list of prediction errors using a model
                trained on the learning set and tested on instances from the
                testing set
        avg_pred_errs -- two-dimensional dictionary with:
            first key corresponding to the name of the learning set,
            second key corresponding to the name of the testing set,
            value corresponding to the average prediction error of the model
                trained on the learning set and tested on instances from the
                testing set
    
    Arguments:
    learner -- scikit-learn estimator
    data1 -- tuple (X, y) representing the first data set, where:
        X -- numpy.array which holds the attribute values
        y -- numpy.array which holds the class value
    data2 -- tuple (X, y) representing the second data set, where:
        X -- numpy.array which holds the attribute values
        y -- numpy.array which holds the class value
    
    """
    # unpack the data1 and data2 tuples
    X1, y1 = data1
    X2, y2 = data2
    # check if both data sets have the same class values
    np.testing.assert_array_equal(np.unique(y1), np.unique(y2),
                err_msg="Both data sets should have the same class values.")
    # data set sizes
    n1 = len(y1)
    n2 = len(y2)
    # prediction errors of models computed as:
    # 1 - P_model(predicted_class == true_class)
    pred_errs1 = []
    pred_errs2 = []
    pred_errsm = []
    
    # first part of leave-one-out (on data1)
    # NOTE: The scikit-learn estimator must be cloned so that each data set
    # gets its own classifier
    model2 = clone(learner)
    model2.fit(X2, y2)
    _check_classes(model2)
    for i in range(n1):
        # current test example
        test_ex = X1[i]
        test_ex_cls = y1[i]
        # create data arrays without the current test example
        cur_data1 = (np.concatenate((X1[:i], X1[(i+1):]), axis=0),
                     np.concatenate((y1[:i], y1[(i+1):]), axis=0))
        cur_datam = (np.concatenate((cur_data1[0], X2), axis=0),
                     np.concatenate((cur_data1[1], y2), axis=0))
        # build models
        model1 = clone(learner)
        model1.fit(*cur_data1)
        _check_classes(model1)
        modelm = clone(learner)
        modelm.fit(*cur_datam)
        _check_classes(modelm)
        # predict the class of test_ex and update prediction errors lists
        probs = model1.predict_proba(test_ex)
        pred_errs1.append(1 - probs[0][test_ex_cls])
        probs = model2.predict_proba(test_ex)
        pred_errs2.append(1 - probs[0][test_ex_cls])
        probs = modelm.predict_proba(test_ex)
        pred_errsm.append(1 - probs[0][test_ex_cls])
    # second part of leave-one-out (on data2)
    # NOTE: The scikit-learn estimator must be cloned so that each data set
    # gets its own classifier
    model1 = clone(learner)
    model1.fit(X1, y1)
    _check_classes(model1)
    for i in range(n2):
        # current test example
        test_ex = X2[i]
        test_ex_cls = y2[i]
        # create data arrays without the current test example
        cur_data2 = (np.concatenate((X2[:i], X2[(i+1):]), axis=0),
                     np.concatenate((y2[:i], y2[(i+1):]), axis=0))
        cur_datam = (np.concatenate((X1, cur_data2[0]), axis=0),
                     np.concatenate((y1, cur_data2[1]), axis=0))
        # build models
        model2 = clone(learner)
        model2.fit(*cur_data2)
        _check_classes(model2)
        modelm = clone(learner)
        modelm.fit(*cur_datam)
        _check_classes(modelm)
        # predict the class of test_ex and update prediction errors lists
        probs = model1.predict_proba(test_ex)
        pred_errs1.append(1 - probs[0][test_ex_cls])
        probs = model2.predict_proba(test_ex)
        pred_errs2.append(1 - probs[0][test_ex_cls])
        probs = modelm.predict_proba(test_ex)
        pred_errsm.append(1 - probs[0][test_ex_cls])
    
    # convert prediction error lists to a two-dimensional dictionary
    pred_errs = {}
    pred_errs["data1"] = {"data1": pred_errs1[:n1], "data2": pred_errs1[n1:],
                          "dataM": pred_errs1}
    pred_errs["data2"] = {"data1": pred_errs2[:n1], "data2": pred_errs2[n1:],
                          "dataM": pred_errs2}
    pred_errs["dataM"] = {"data1": pred_errsm[:n1], "data2": pred_errsm[n1:],
                          "dataM": pred_errsm}
    
    return pred_errs, _compute_average_prediction_errors(pred_errs)

def _generalized_cross_validation(learner, data1, data2, cv_folds1):
    """Perform one part of the generalized version of the cross-validation
    testing method on the given data sets.
    Perform cross-validation over data set data1. For each fold of data1,
    build models on the remaining folds of data1, the whole data set data2 and
    the merged data set and test them on the selected fold of data1.
    Return a tuple (pred_errs1, pred_errs2, pred_errsm), where:
        pred_errs1 -- numpy.array of prediction errors of the model built on the
            remaining folds of data1 for instances in data1
        pred_errs2 -- numpy.array of prediction errors of the model built on the
            whole data set data2 for instances in data1
        pred_errm -- numpy.array of prediction errors of the model built on the
            merged data set for instances in data1
    
    Arguments:
    learner -- scikit-learn estimator
    data1 -- tuple (X, y) representing the first data set, where:
        X -- numpy.array which holds the attribute values
        y -- numpy.array which holds the class value
    data2 -- tuple (X, y) representing the second data set, where:
        X -- numpy.array which holds the attribute values
        y -- numpy.array which holds the class value
    cv_folds1 -- list of tuples (learn, test) to perform cross-validation over
        data1, where:
        learn -- numpy.array with a Boolean mask for selecting learning
            instances
        test -- numpy.array with a Boolean mask for selecting testing instances
    
    """
    # unpack the data1 and data2 tuples
    X1, y1 = data1
    X2, y2 = data2
    # build a model on data2
    # NOTE: The model does not change throughout cross-validation on data1
    # NOTE: When the number of unique class values is less than 2, we
    # cannot fit an ordinary model (e.g. logistic regression). Instead, we
    # have to use a dummy classifier which is subsequently augmented to
    # handle all the other class values.
    # NOTE: The scikit-learn estimator must be cloned so that each data set
    # gets its own classifier
    if len(np.unique(y2)) < 2:
        model2 = DummyClassifier()
        model2.fit(X2, y2)
        change_dummy_classes(model2, np.array([0, 1]))
    else:
        model2 = clone(learner)
        model2.fit(X2, y2)
    _check_classes(model2)
    # prediction errors of models computed as:
    # 1 - P_model(predicted_class == true_class)
    # (pred. errors of the model built on data2 can be computed right away) 
    pred_proba2 = model2.predict_proba(X1)
    pred_errs2 = 1 - pred_proba2[np.arange(y1.shape[0]), y1]
    pred_errs1 = -np.ones(y1.shape)
    pred_errsm = -np.ones(y1.shape)
    # perform generalized cross-validation on data1
    for learn_ind, test_ind in cv_folds1:
        # create testing data arrays for the current fold
        test_X, test_y = X1[test_ind], y1[test_ind]
        # create learning data arrays for the current fold
        learn1 = X1[learn_ind], y1[learn_ind]
        learnm = (np.concatenate((X1[learn_ind], X2), axis=0),
                  np.concatenate((y1[learn_ind], y2), axis=0))
        # build models
        # NOTE: When the number of unique class values is less than 2, we
        # cannot fit an ordinary model (e.g. logistic regression). Instead, we
        # have to use a dummy classifier which is subsequently augmented to
        # handle all the other class values.
        # NOTE: The scikit-learn estimator must be cloned so that each data
        # set gets its own classifier 
        if len(np.unique(learn1[1])) < 2:
            model1 = DummyClassifier()
            model1.fit(*learn1)
            change_dummy_classes(model1, np.array([0, 1]))
        else:
            model1 = clone(learner)
            model1.fit(*learn1)
        _check_classes(model1)
        if len(np.unique(learnm[1])) < 2:
            modelm = DummyClassifier()
            modelm.fit(*learn1)
            change_dummy_classes(modelm, np.array([0, 1]))
        else:
            modelm = clone(learner)
            modelm.fit(*learnm)
        _check_classes(modelm)
        # compute the prediction errors of both models on the current testing
        # data
        pred_proba1 = model1.predict_proba(test_X)
        pred_errs1[test_ind] = 1 - pred_proba1[np.arange(test_y.shape[0]),
                                               test_y]
        pred_probam = modelm.predict_proba(test_X)
        pred_errsm[test_ind] = 1 - pred_probam[np.arange(test_y.shape[0]),
                                               test_y]
    return pred_errs1, pred_errs2, pred_errsm

def generalized_cross_validation(learner, data1, data2, folds, rand_seed1,
                                 rand_seed2):
    """Perform a generalized version of the cross-validation testing method on
    the given data sets.
    Estimate the prediction errors of models built on all combinations of the
    data sets (data1, data2 and merged data set) and tested across all
    combinations of the data sets (data1, data2 and merged data set).
    Return a tuple (pred_errs, avg_pred_errs), where:
        pred_errs -- two-dimensional dictionary with:
            first key corresponding to the name of the learning set,
            second key corresponding to the name of the testing set,
            value corresponding to the numpy.array of prediction errors using a
                model trained on the learning set and tested on instances from
                the testing set
        avg_pred_errs -- two-dimensional dictionary with:
            first key corresponding to the name of the learning set,
            second key corresponding to the name of the testing set,
            value corresponding to the average prediction error of the model
                trained on the learning set and tested on instances from the
                testing set
    
    Arguments:
    learner -- scikit-learn estimator
    data1 -- tuple (X, y) representing the first data set, where:
        X -- numpy.array which holds the attribute values
        y -- numpy.array which holds the class value
    data2 -- tuple (X, y) representing the second data set, where:
        X -- numpy.array which holds the attribute values
        y -- numpy.array which holds the class value
    folds -- integer representing the number of folds
    rand_seed1 -- integer used as the random seed for creating cross-validation
        random folds for data1
    rand_seed2 -- integer used as the random seed for creating cross-validation
        random folds for data2
    
    """
    # unpack the data1 and data2 tuples
    _, y1 = data1
    _, y2 = data2
    # create a two-dimensional dictionary storing prediction error lists of
    # models build on all combinations of the learning data sets and
    # tested on all combinations of the testing data sets
    pred_errs = {"data1" : {}, "data2": {}, "dataM": {}}
    # generate cross-validation folds
    cv_folds1 = cross_validation.KFold(len(y1), folds, indices=False,
                                       shuffle=True, random_state=rand_seed1)
    cv_folds2 = cross_validation.KFold(len(y2), folds, indices=False,
                                       shuffle=True, random_state=rand_seed2)
    # first part of cross-validation (on data1)
    pred_errs1, pred_errs2, pred_errsm = _generalized_cross_validation(learner,
                                            data1, data2, cv_folds1)
    pred_errs["data1"]["data1"] = pred_errs1
    pred_errs["data2"]["data1"] = pred_errs2
    pred_errs["dataM"]["data1"] = pred_errsm
    # second part of cross-validation (on data2)
    pred_errs2, pred_errs1, pred_errsm = _generalized_cross_validation(learner,
                                            data2, data1, cv_folds2)
    pred_errs["data1"]["data2"] = pred_errs1
    pred_errs["data2"]["data2"] = pred_errs2
    pred_errs["dataM"]["data2"] = pred_errsm
    # "compute" prediction errors for the merged data set
    for data in ("data1", "data2", "dataM"):
        pred_errs[data]["dataM"] = np.concatenate((pred_errs[data]["data1"],
                                        pred_errs[data]["data2"]), axis=0)
    return pred_errs, _compute_average_prediction_errors(pred_errs)

if __name__ == "__main__":
    import random
    random.seed(17)
    
    # CHOOSE LEARNING ALGORITHM
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    
    # LOAD DATA
    from PyMTL import data
    tasks = data.load_usps_digits_data()
    data1 = tasks[0].data, tasks[0].target
    data2 = tasks[1].data, tasks[1].target
    print "data1 has {} examples, data2 has {} examples".format(len(data1[0]),
                                                                len(data2[0]))
    # SET PARAMETERS
    folds = 5
    rand_seed1 = random.randint(0, 100)
    rand_seed2 = random.randint(0, 100)

    # VIEW RESULTS
    _, avg_errs_cv = generalized_cross_validation(clf, data1, data2, folds,
                                                  rand_seed1, rand_seed2)

    for learn_data in ("data1", "data2", "dataM"):
        for test_data in ("data1", "data2", "dataM"):
            print "Learn: {}, Test: {}, Avg. pred. err.: {}".format(
                learn_data, test_data, avg_errs_cv[learn_data][test_data])

    
#    # COMPARE PERFORMANCE
#    # GENERALIZED LEAVE-ONE-OUT
#    from timeit import Timer
#    t = Timer("generalized_leave_one_out(learner, data1, data2)", "gc.enable();"
#              " from __main__ import generalized_leave_one_out, learner, data1,"
#              " data2")
#    repeats = 10
#    elapsed = t.timeit(repeats)
#    print "Average time it took to complete one call to the generalized " \
#        "leave-one-out() method: {:.3f}s ({} repetitions)".format(
#        elapsed/repeats, repeats)
#    # GENERALIZED CROSS-VALIDATION
#    from timeit import Timer
#    t = Timer("generalized_cross_validation(learner, data1, data2, folds, "
#              "rand_seed1, rand_seed2)", "gc.enable(); from __main__ import "
#              "generalized_cross_validation, learner, data1, data2, folds, "
#              "rand_seed1, rand_seed2")
#    repeats = 10
#    elapsed = t.timeit(repeats)
#    print "Average time it took to complete one call to the generalized " \
#        "cross-validation() method: {:.3f}s ({} repetitions)".format(
#        elapsed/repeats, repeats)

#    # COMPARE RESULTS
#    _, avg_errs_loo = generalized_leave_one_out(clf, data1, data2)
#    _, avg_errs_cv = generalized_cross_validation(clf, data1, data2, folds,
#                                                  rand_seed1, rand_seed2)
#    for learn_data in ("data1", "data2", "dataM"):
#        for test_data in ("data1", "data2", "dataM"):
#            print "Learn: {}, Test: {}, Difference (LOO - CV): {}".format(
#                learn_data, test_data, avg_errs_loo[learn_data][test_data] - \
#                avg_errs_cv[learn_data][test_data])
