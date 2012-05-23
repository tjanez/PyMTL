#
# testing.py
# Contains classes and methods for internal testing used by merging learning
# methods.
#
# Copyright (C) 2011, 2012 Tadej Janez
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

import numpy

import Orange

def _compute_average_prediction_errors(pred_errs):
    """Compute average prediction errors from the given prediction error lists.
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
        value corresponding to the list of prediction errors using a model
            trained on the learning set and tested on instances from the
            testing set
    
    """
    avg_pred_errs = {}
    for learn_data in pred_errs.iterkeys():
        avg_pred_errs[learn_data] = {}
        for test_data in pred_errs[learn_data].iterkeys():
            avg_pred_errs[learn_data][test_data] = \
                numpy.mean(pred_errs[learn_data][test_data])
    return avg_pred_errs

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
    learner -- Orange learner
    data1 -- Orange data table representing the first data set
    data2 -- Orange data table representing the second data set
    
    """
    # check if both domains have the same class values
    if data1.domain.class_var.values != data2.domain.class_var.values:
        raise ValueError("Both data sets should have the same class values.")
    # data set sizes
    n1 = len(data1)
    n2 = len(data2)
    # prediction errors of models computed as:
    # 1 - P_model(predicted_class == true_class)
    pred_errs1 = []
    pred_errs2 = []
    pred_errsm = []
    
    # first part of leave-one-out (on data1)
    model2 = learner(data2)
    for i in range(n1):
        # current test example
        test_ex = Orange.data.Instance(data1[i])
        test_ex_cls = test_ex.get_class()
        test_ex.set_class('?')
        # create data tables without the current test example
        cur_data1 = Orange.data.Table(data1[:i] + data1[(i+1):])
        cur_datam = Orange.data.Table(cur_data1[:] + data2[:])
        # build models
        model1 = learner(cur_data1)
        modelm = learner(cur_datam)
        # predict the class of test_ex and update prediction errors lists
        prob = Orange.classification.Classifier.GetProbabilities
        probs = model1(test_ex, result_type=prob)
        pred_errs1.append(1 - probs[test_ex_cls])
        probs = model2(test_ex, result_type=prob)
        pred_errs2.append(1 - probs[test_ex_cls])
        probs = modelm(test_ex, result_type=prob)
        pred_errsm.append(1 - probs[test_ex_cls])
    # second part of leave-one-out (on data2)
    model1 = learner(data1)
    for i in range(n2):
        # current test example
        test_ex = Orange.data.Instance(data2[i])
        test_ex_cls = test_ex.get_class()
        test_ex.set_class('?')
        # create data sets without the current test example
        cur_data2 = Orange.data.Table(data2[:i] + data2[(i+1):])
        cur_datam = Orange.data.Table(data1[:] + cur_data2[:])
        # build models
        model2 = learner(cur_data2)
        modelm = learner(cur_datam)
        # predict the class of test_ex and update prediction errors lists
        prob = Orange.classification.Classifier.GetProbabilities
        probs = model1(test_ex, result_type=prob)
        pred_errs1.append(1 - probs[test_ex_cls])
        probs = model2(test_ex, result_type=prob)
        pred_errs2.append(1 - probs[test_ex_cls])
        probs = modelm(test_ex, result_type=prob)
        pred_errsm.append(1 - probs[test_ex_cls])
    
    # convert prediction error lists to a two-dimensional dictionary
    pred_errs = {}
    pred_errs["data1"] = {"data1": pred_errs1[:n1], "data2": pred_errs1[n1:],
                          "dataM": pred_errs1}
    pred_errs["data2"] = {"data1": pred_errs2[:n1], "data2": pred_errs2[n1:],
                          "dataM": pred_errs2}
    pred_errs["dataM"] = {"data1": pred_errsm[:n1], "data2": pred_errsm[n1:],
                          "dataM": pred_errsm}
    
    return pred_errs, _compute_average_prediction_errors(pred_errs)

def _generalized_cross_validation(learner, data1, data2, cv_indices1):
    """Perform one part of the generalized version of the cross-validation
    testing method on the given data sets.
    Perform cross-validation over data set data1. For each fold of data1,
    build models on the remaining folds of data1, the whole data set data2 and
    the merged data set and test them on the selected fold of data1.
    Return a tuple (pred_errs1, pred_errs2, pred_errsm), where:
        pred_errs1 -- list of prediction errors of the model built on the
            remaining folds of data1 for instances in data1
        pred_errs2 -- list of prediction errors of the model built on the whole
            data set data2 for instances in data1
        pred_errm -- list of prediction errors of the model built on the merged
            data set for instances in data1
    
    Arguments:
    learner -- Orange learner
    data1 -- Orange data table representing the first data set
    data2 -- Orange data table representing the second data set
    cv_indices1 -- Orange LongList with indices for performing cross-validation
        over data1
    
    """
    n1 = len(data1)
    folds = max(cv_indices1) + 1
    # prediction errors of models computed as:
    # 1 - P_model(predicted_class == true_class)
    pred_errs1 = [None]*n1
    pred_errs2 = [None]*n1
    pred_errsm = [None]*n1
    # build a model on data2
    # Note: The model does not change throughout cross-validation on data1
    model2 = learner(data2)
    # perform generalized cross-validation on data1
    for fold in range(folds):
        # create testing data table for the current fold
        test = data1.select_ref(cv_indices1, fold)
        test_ids = [i for i, _ in enumerate(data1) if cv_indices1[i] == fold]
        # create learning data tables for the current fold
        learn1 = data1.select_ref(cv_indices1, fold, negate=True)
        learnm = Orange.data.Table(learn1[:] + data2[:])
        # build models
        model1 = learner(learn1)
        modelm = learner(learnm)
        # test models on examples in the testing data table
        for ex, id in zip(test, test_ids):
            # hide actual class to prevent cheating
            test_ex = Orange.data.Instance(ex)
            test_ex_cls = test_ex.get_class()
            test_ex.set_class('?')
            # predict the class of test_ex and update prediction errors lists
            prob = Orange.classification.Classifier.GetProbabilities
            probs = model1(test_ex, result_type=prob)
            pred_errs1[id] = (1 - probs[test_ex_cls])
            probs = model2(test_ex, result_type=prob)
            pred_errs2[id] = (1 - probs[test_ex_cls])
            probs = modelm(test_ex, result_type=prob)
            pred_errsm[id] = (1 - probs[test_ex_cls])
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
    learner -- Orange learner
    data1 -- Orange data table representing the first data set
    data2 -- Orange data table representing the second data set
    folds -- integer representing the number of folds
    rand_seed1 -- integer used as the random seed for creating cross-validation
        random indices for data1
    rand_seed2 -- integer used as the random seed for creating cross-validation
        random indices for data2
    
    """
    # check if both domains have the same class values
    if data1.domain.class_var.values != data2.domain.class_var.values:
        raise ValueError("Both data sets should have the same class values.")
    # create a two-dimensional dictionary storing prediction error lists of
    # models build on all combinations of the learning data sets and
    # tested on all combinations of the testing data sets
    pred_errs = {"data1" : {}, "data2": {}, "dataM": {}}
    # generate random indices
    strat_if_pos = Orange.core.MakeRandomIndices.StratifiedIfPossible
    cv_indices1 = Orange.core.MakeRandomIndicesCV(data1, folds, stratified=\
                                strat_if_pos, randseed=rand_seed1)
    cv_indices2 = Orange.core.MakeRandomIndicesCV(data2, folds, stratified=\
                                strat_if_pos, randseed=rand_seed2)
    # first part of cross-validation (on data1)
    pred_errs1, pred_errs2, pred_errsm = _generalized_cross_validation(learner,
                                            data1, data2, cv_indices1)
    pred_errs["data1"]["data1"] = pred_errs1
    pred_errs["data2"]["data1"] = pred_errs2
    pred_errs["dataM"]["data1"] = pred_errsm
    # second part of cross-validation (on data2)
    pred_errs2, pred_errs1, pred_errsm = _generalized_cross_validation(learner,
                                            data2, data1, cv_indices2)
    pred_errs["data1"]["data2"] = pred_errs1
    pred_errs["data2"]["data2"] = pred_errs2
    pred_errs["dataM"]["data2"] = pred_errsm
    # "compute" prediction errors for the merged data set
    for data in ("data1", "data2", "dataM"):
        pred_errs[data]["dataM"] = pred_errs[data]["data1"] + \
                                    pred_errs[data]["data2"]
    return pred_errs, _compute_average_prediction_errors(pred_errs)

if __name__ == "__main__":
    import random
    random.seed(17)
    
    learner = Orange.classification.bayes.NaiveLearner()
#     # users with ~1100 ratings
#    data1_file = "/home/tadej/Workspace/ERMRec/data/users-m100/user00689.tab"
#    data2_file = "/home/tadej/Workspace/ERMRec/data/users-m100/user00559.tab"
    # users with ~50 ratings
    data1_file = "/home/tadej/Workspace/ERMRec/data/users-m50/user04752.tab"
    data2_file = "/home/tadej/Workspace/ERMRec/data/users-m50/user00663.tab"
#    # users with 10-20 ratings
#    data1_file = "/home/tadej/Workspace/ERMRec/data/users-test/user00009.tab"
#    data2_file = "/home/tadej/Workspace/ERMRec/data/users-test/user00017.tab"
    data1 = Orange.data.Table(data1_file)
    data2 = Orange.data.Table(data2_file)
    print "data1 has {} examples, data2 has {} examples".format(len(data1),
                                                                len(data2))
    
    # GENERALIZED LEAVE-ONE-OUT
#    pred_errs, avg_pred_errs = generalized_leave_one_out(learner, data1, data2)
#    from timeit import Timer
#    t = Timer("generalized_leave_one_out(learner, data1, data2)",
#            "gc.enable(); from __main__ import generalized_leave_one_out, "\
#            "learner, data1, data2")
#    repeats = 10
#    elapsed = t.timeit(repeats)
#    print "Average time it took to complete one call to the generalized " \
#        "leave-one-out() method: {:.3f}s ({} repetitions)".format(
#        elapsed/repeats, repeats)
        
    # GENERALIZED CROSS-VALIDATION
    folds = 5
    rand_seed1 = random.randint(0, 100)
    rand_seed2 = random.randint(0, 100)
#    pred_errs, avg_pred_errs = generalized_cross_validation(learner, data1,
#                                    data2, folds, rand_seed1, rand_seed2)
#    from timeit import Timer
#    t = Timer("generalized_cross_validation(learner, data1, data2, folds, " \
#            "rand_seed1, rand_seed2)",
#            "gc.enable(); from __main__ import generalized_cross_validation, "\
#            "learner, data1, data2, folds, rand_seed1, rand_seed2")
#    repeats = 10
#    elapsed = t.timeit(repeats)
#    print "Average time it took to complete one call to the generalized " \
#        "cross-validation() method: {:.3f}s ({} repetitions)".format(
#        elapsed/repeats, repeats)

    # COMPARE RESULTS
    _, avg_errs_loo = generalized_leave_one_out(learner, data1, data2)
    _, avg_errs_cv = generalized_cross_validation(learner, data1, data2, folds,
                                                  rand_seed1, rand_seed2)
    for learn_data in ("data1", "data2", "dataM"):
        for test_data in ("data1", "data2", "dataM"):
            print "Learn: {}, Test: {}, Difference (LOO - CV): {}".format(
                learn_data, test_data, avg_errs_loo[learn_data][test_data] - \
                avg_errs_cv[learn_data][test_data])
    