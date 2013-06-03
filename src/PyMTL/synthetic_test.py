#
# synthetic_test.py
# Extends the test module for performing tests on synthetic MTL problems.
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

import logging, os.path, time
from collections import OrderedDict

import numpy as np

from PyMTL import synthetic_data
from PyMTL.util import logger, configure_logger
from PyMTL.learning import prefiltering, learning
from PyMTL.test import test_tasks


if __name__ == "__main__":
    # find out the current file's location so it can be used to compute the
    # location of other files/directories
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    
    # base learners for Boolean functions
    base_learners_bool = OrderedDict()
    from sklearn.tree import DecisionTreeClassifier
    base_learners_bool["tree"] = DecisionTreeClassifier()
    from sklearn.svm import SVC
    base_learners_bool["svm_poly"] = SVC(kernel="poly", coef0=1, degree=5,
                                         probability=True)
    
    # scoring measures for classification problems
    measures_clas = []
    measures_clas.append("CA")
    measures_clas.append("AUC")
    
    learners = OrderedDict()
    learners["NoMerging"] = learning.NoMergingLearner()
    learners["MergeAll"] = learning.MergeAllLearner()
    no_filter = prefiltering.NoFilter()
    learners["ERM"] = learning.ERMLearner(folds=5, seed=33, prefilter=no_filter,
                                          error_func=None)
    
    # boolean indicating which testing configuration to use:
    # 1 -- Boolean function (toy problem)
    test_config = 1
    
    # boolean indicating whether to perform the tests on the MTL problem
    test = True
    # boolean indicating whether to find previously computed testing results
    # and unpickling them
    unpickle = False
    # boolean indicating whether to visualize the results of the MTL problem
    visualize = True
    
    if test_config == 1:
        # parameters of the synthetic Boolean MTL problem
        attributes = 8
        disjunct_degree = 4
        n = 200
        task_groups = 2
        tasks_per_group = 5
        noise = 0.0
        data_rnd_seed = 11
        # parameters of the MTL problem tester
        rnd_seed = 51
        repeats = 3
        test_prop=0.5
        results_path = os.path.join(path_prefix, "results/synthetic_data/"
                        "boolean_func-a{}d{}n{}g{}tg{}-seed{}-repeats{}".\
                        format(attributes, disjunct_degree, n, task_groups,
                               tasks_per_group, rnd_seed, repeats))
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
        configure_logger(logger, console_level=logging.INFO, file_name=log_file)
        tasks_data = synthetic_data.generate_boolean_data(attributes,
                        disjunct_degree, n, task_groups, tasks_per_group, noise,
                        random_seed=data_rnd_seed)
        test_tasks(tasks_data, results_path, base_learners_bool,
                   measures_clas, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=test_prop, repeats=repeats, cfg_logger=False)
