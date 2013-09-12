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

from PyMTL import synthetic_data
from PyMTL.util import logger, configure_logger, unpickle_obj
from PyMTL.learning import prefiltering, learning
from PyMTL.test import test_tasks, log_base_learner_info, LEARNERS_TO_COLORS
from PyMTL.plotting import LinePlotDesc, plot_multiple_separate
from PyMTL.stat import convert_std_to_ci95


def combine_experiment_results(results_path_fmt, chan_par_values,
        file_name_fmt, repeats, error_measure="std", title="", xlabel=""):
    """Combine the results of experiments matching the given results_path_fmt,
    where the value of a particular parameter changes according to the given
    chan_par_values list.
    Draw a plot showing the combined results and save it to the given file name
    pattern.
    
    Parameters
    ----------
    results_path_fmt : string
        Template for directory names containing pickled results. It must contain
        exactly one pair of braces ({}) where the value of the changing
        parameter will be put.
    chan_par_values : list
        A list of values of the changing parameter (i.e. the parameter, whose
        value is not fixed during this experiment).
    file_name_fmt : string
        Template for the paths where to save the drawn plots. It must contain
        exactly one pair of braces ({}) where the base learner's name will be
        put.
    repeats : int
        Number of repetitions of each experiment.
    error_measure : string
        Indicator of which measure to use for plots' error bars.
        Currently, only "std" (standard deviation) and "95ci" (95% confidence
        intervals) are supported.
    title : string (optinal)
        The title of each plot.
    xlabel : string (optinal)
        Each plot's x-axis label.
    
    """
    # key for extracting the AUC results from the pickled dictionaries
    AUC_STRING = ("Results for AUC (weighting method: all_equal, error margin "
                  "measure: std)")
    
    ERROR_MEASURE_TO_PRINT = {"std" : "std. dev.",
                              "95ci" : "95% conf. intervals"}
    
    def _convert_error(err):
        """Convert the given error value according to the error_measure
        parameter.
        
        """
        if error_measure == "std":
            return err
        elif error_measure == "95ci":
            return convert_std_to_ci95(err, repeats)
        else:
            raise ValueError("Unknown error measure: {}".format(error_measure))
    
    res = dict()
    # unpickle the results
    for v in chan_par_values:
        results_path = results_path_fmt.format(v)
        res[v] = unpickle_obj(os.path.join(results_path, "overall_results.pkl"))
    
    # extract the names of base learners and learners
    base_learners = res[chan_par_values[0]][AUC_STRING].keys()
    learners = res[chan_par_values[0]][AUC_STRING][base_learners[0]].keys()
    
    # transform the results to match the input of the plot_multiple_separate()
    # function
    x_points = OrderedDict()
    avgs = OrderedDict()
    errs = OrderedDict()
    for v in chan_par_values:
        res_auc = res[v][AUC_STRING]
        for bl in base_learners:
            if bl not in x_points:
                x_points[bl] = OrderedDict()
            if bl not in avgs:
                avgs[bl] = OrderedDict()
            if bl not in errs:
                errs[bl] = OrderedDict()
            for l in learners:
                if l not in x_points[bl]:
                    x_points[bl][l] = []
                if l not in avgs[bl]:
                    avgs[bl][l] = []
                if l not in errs[bl]:
                    errs[bl][l] = []
                x_points[bl][l].append(v)
                avg, err = res_auc[bl][l]
                avgs[bl][l].append(avg)
                errs[bl][l].append(_convert_error(err))
    
    # generate the plot description objects
    plot_desc = OrderedDict()
    for bl in base_learners:
        plot_desc[bl] = []
        for l in learners:
            plot_desc[bl].append(LinePlotDesc(x_points[bl][l], avgs[bl][l],
                errs[bl][l], l, color=LEARNERS_TO_COLORS[l],
                ecolor=LEARNERS_TO_COLORS[l]))
    
    # draw and save the plots
    
    if title == "":
        title = "Error bars show {}".format(ERROR_MEASURE_TO_PRINT[
                                            error_measure])
    else:
        title = title + " (error bars show {})".format(ERROR_MEASURE_TO_PRINT[
                                                        error_measure])
    plot_multiple_separate(plot_desc, file_name_fmt,
        title=title, xlabel=xlabel, ylabel="AUC",
        x_tick_points=chan_par_values,
        ylim_bottom=0, ylim_top=1, error_bars=True)


if __name__ == "__main__":
    # find out the current file's location so it can be used to compute the
    # location of other files/directories
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    
    # base learners for Boolean functions
    base_learners_bool = OrderedDict()
#    from sklearn.tree import DecisionTreeClassifier
#    base_learners_bool["tree"] = DecisionTreeClassifier()
    from sklearn.svm import SVC
    base_learners_bool["svm_poly3"] = SVC(kernel="poly", coef0=1, degree=3,
                                          probability=True)
#    base_learners_bool["svm_poly5"] = SVC(kernel="poly", coef0=1, degree=5,
#                                          probability=True)
#    from PyMTL.orange_utils import OrangeClassifierWrapper
#    from Orange.classification.tree import TreeLearner
#    base_learners_bool["orange_tree_pruned"] = OrangeClassifierWrapper(
#        TreeLearner(min_instances=10, same_majority_pruning=True,
#                    store_instances=True))
    
    
    # scoring measures for classification problems
    measures_clas = []
    measures_clas.append("CA")
    measures_clas.append("AUC")
    
    learners = OrderedDict()
    no_filter = prefiltering.NoFilter()
    learners["ERM"] = learning.ERMLearner(folds=5, seed=33, prefilter=no_filter,
                                          error_func=None)
    learners["NoMerging"] = learning.NoMergingLearner()
    learners["MergeAll"] = learning.MergeAllLearner()
    learners["Oracle"] = learning.OracleLearner(r"^Group (\d+), task \d+$")
    
    # list of integers indicating which testing configuration to use:
    # 1 -- Boolean problem (toy problem with a = 8 and d = 4)
    # 2 -- Boolean problem (2 groups, 5 tasks per group)
    # 3 -- Boolean problem (2 groups, 10 tasks per group)
    # 4 -- Boolean problem (5 groups, 5 tasks per group)
    # 5 -- Boolean problem (5 groups, 5 tasks per group; comparison of different
    #                       polynomial SVMs)
    # 6 -- Boolean problem (10 groups, 10 tasks per group)
    test_configs = [41, 42, 43, 44]
    
    # list of strings indicating in which mode the script is to be run
    # (only applies to testing configurations 41 -- 44):
    # run -- run the experiments
    # combine -- combine the results of the experiments
    mode = ["combine"]
    
    # string indicating the measure to use when plotting error bars:
    # std -- standard deviation
    # 95ci -- 95% confidence intervals
    error_measure = "95ci"
    
    # boolean indicating whether to perform the tests on the MTL problem
    test = True
    # boolean indicating whether to find previously computed testing results
    # and unpickling them
    unpickle = False
    # boolean indicating whether to visualize the results of the MTL problem
    visualize = True
    
    for test_config in test_configs:
        
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
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            tasks_data = synthetic_data.generate_boolean_data(attributes,
                            disjunct_degree, n, task_groups, tasks_per_group,
                            noise, random_seed=data_rnd_seed)
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "train_test_split",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       test_prop=test_prop, repeats=repeats, cfg_logger=False,
                       separate_figs=True)
        
        if test_config == 21:
            # parameters of the synthetic Boolean MTL problem
            attributes = 8
            disjunct_degree = 4
            n = 100
            task_groups = 2
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 11
            n_learning_sets = 3
            # parameters of the MTL problem tester
            rnd_seed = 51
            # prepare directories and loggers
            results_path = (os.path.join(path_prefix, "results/synthetic_data/"
                "bool_func-a{}d{}n{}g{}tg{}nse{}rs{}nls{}-seed{}-complete_test"
                "".format(attributes, disjunct_degree, n, task_groups,
                tasks_per_group, noise, data_rnd_seed, n_learning_sets,
                rnd_seed)))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            log_base_learner_info(logger, base_learners_bool)
            # generate boolean data with complete test sets
            funcs_pickle_path = os.path.join(results_path, "boolean_funcs.pkl")
            tasks_data, tasks_complete_test_sets = \
                synthetic_data.generate_boolean_data_with_complete_test_sets(
                    attributes, disjunct_degree, n, task_groups,
                    tasks_per_group, noise, random_seed=data_rnd_seed,
                    n_learning_sets=n_learning_sets,
                    funcs_pickle_path=funcs_pickle_path)
            # test the generated MTL problem
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "pre-prepared_test",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       preprepared_test_sets=tasks_complete_test_sets,
                       separate_figs=True, cfg_logger=False)
        
        if test_config == 2:
            # parameters of the synthetic Boolean MTL problem
            attributes = 16
            disjunct_degree = 8
            n = 200
            task_groups = 2
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            # parameters of the MTL problem tester
            rnd_seed = 51
            repeats = 3
            test_prop=0.5
            results_path = os.path.join(path_prefix, "results/synthetic_data/"
                "boolean_func-a{}d{}n{}g{}tg{}rs{}-seed{}-repeats{}".\
                format(attributes, disjunct_degree, n, task_groups,
                       tasks_per_group, data_rnd_seed, rnd_seed, repeats))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            tasks_data = synthetic_data.generate_boolean_data(attributes,
                            disjunct_degree, n, task_groups, tasks_per_group,
                            noise, random_seed=data_rnd_seed)
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "train_test_split",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       test_prop=test_prop, repeats=repeats, cfg_logger=False,
                       separate_figs=True)
        
        if test_config == 22:
            # parameters of the synthetic Boolean MTL problem
            attributes = 16
            disjunct_degree = 8
            n = 100
            task_groups = 2
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            n_learning_sets = 3
            # parameters of the MTL problem tester
            rnd_seed = 51
            # prepare directories and loggers
            results_path = (os.path.join(path_prefix, "results/synthetic_data/"
                "bool_func-a{}d{}n{}g{}tg{}nse{}rs{}nls{}-seed{}-complete_test"
                "".format(attributes, disjunct_degree, n, task_groups,
                tasks_per_group, noise, data_rnd_seed, n_learning_sets,
                rnd_seed)))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            log_base_learner_info(logger, base_learners_bool)
            # generate boolean data with complete test sets
            funcs_pickle_path = os.path.join(results_path, "boolean_funcs.pkl")
            tasks_data, tasks_complete_test_sets = \
                synthetic_data.generate_boolean_data_with_complete_test_sets(
                    attributes, disjunct_degree, n, task_groups,
                    tasks_per_group, noise, random_seed=data_rnd_seed,
                    n_learning_sets=n_learning_sets,
                    funcs_pickle_path=funcs_pickle_path)
            # test the generated MTL problem
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "pre-prepared_test",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       preprepared_test_sets=tasks_complete_test_sets,
                       separate_figs=True, cfg_logger=False)
    
        if test_config == 3:
            # parameters of the synthetic Boolean MTL problem
            attributes = 16
            disjunct_degree = 8
            n = 200
            task_groups = 2
            tasks_per_group = 10
            noise = 0.0
            data_rnd_seed = 12
            # parameters of the MTL problem tester
            rnd_seed = 51
            repeats = 3
            test_prop=0.5
            results_path = os.path.join(path_prefix, "results/synthetic_data/"
                "boolean_func-a{}d{}n{}g{}tg{}rs{}-seed{}-repeats{}".\
                format(attributes, disjunct_degree, n, task_groups,
                       tasks_per_group, data_rnd_seed, rnd_seed, repeats))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            tasks_data = synthetic_data.generate_boolean_data(attributes,
                            disjunct_degree, n, task_groups, tasks_per_group,
                            noise, random_seed=data_rnd_seed)
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "train_test_split",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       test_prop=test_prop, repeats=repeats, cfg_logger=False,
                       separate_figs=True)
        
        if test_config == 4:
            # parameters of the synthetic Boolean MTL problem
            attributes = 16
            disjunct_degree = 8
            n = 200
            task_groups = 5
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            # parameters of the MTL problem tester
            rnd_seed = 51
            repeats = 3
            test_prop=0.5
            results_path = os.path.join(path_prefix, "results/synthetic_data/"
                "boolean_func-a{}d{}n{}g{}tg{}rs{}-seed{}-repeats{}".\
                format(attributes, disjunct_degree, n, task_groups,
                       tasks_per_group, data_rnd_seed, rnd_seed, repeats))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            tasks_data = synthetic_data.generate_boolean_data(attributes,
                            disjunct_degree, n, task_groups, tasks_per_group,
                            noise, random_seed=data_rnd_seed)
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "train_test_split",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       test_prop=test_prop, repeats=repeats, cfg_logger=False,
                       separate_figs=True)
        
        if test_config == 5:
            # parameters of the synthetic Boolean MTL problem
            attributes = 16
            disjunct_degree = 8
            n = 200
            task_groups = 5
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            # parameters of the MTL problem tester
            rnd_seed = 51
            repeats = 3
            test_prop=0.5
            # different polynomial SVMs
            # base learners for Boolean functions
            base_learners_poly = OrderedDict()
            from sklearn.svm import SVC
            for d in [3, 5, 10]:
                base_learners_poly["svm_poly_deg{}".format(d)] = SVC(
                    kernel="poly", coef0=1, degree=d, probability=True)
            results_path = os.path.join(path_prefix, "results/synthetic_data/"
                "boolean_func-a{}d{}n{}g{}tg{}rs{}-seed{}-repeats{}"
                "-svm_poly_comp".format(attributes, disjunct_degree, n,
                    task_groups, tasks_per_group, data_rnd_seed, rnd_seed,
                    repeats))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            tasks_data = synthetic_data.generate_boolean_data(attributes,
                            disjunct_degree, n, task_groups, tasks_per_group,
                            noise, random_seed=data_rnd_seed)
            test_tasks(tasks_data, results_path, base_learners_poly,
                       measures_clas, learners, "train_test_split",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       test_prop=test_prop, repeats=repeats, cfg_logger=False,
                       separate_figs=True)
    
        if test_config == 6:
            # parameters of the synthetic Boolean MTL problem
            attributes = 16
            disjunct_degree = 8
            n = 200
            task_groups = 10
            tasks_per_group = 10
            noise = 0.0
            data_rnd_seed = 12
            # parameters of the MTL problem tester
            rnd_seed = 52
            repeats = 3
            test_prop=0.5
            results_path = os.path.join(path_prefix, "results/synthetic_data/"
                "boolean_func-a{}d{}n{}g{}tg{}rs{}-seed{}-repeats{}".\
                format(attributes, disjunct_degree, n, task_groups,
                       tasks_per_group, data_rnd_seed, rnd_seed, repeats))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
            configure_logger(logger, console_level=logging.INFO,
                             file_name=log_file)
            tasks_data = synthetic_data.generate_boolean_data(attributes,
                            disjunct_degree, n, task_groups, tasks_per_group,
                            noise, random_seed=data_rnd_seed)
            test_tasks(tasks_data, results_path, base_learners_bool,
                       measures_clas, learners, "train_test_split",
                       rnd_seed=rnd_seed,
                       test=test, unpickle=unpickle, visualize=visualize,
                       test_prop=test_prop, repeats=repeats, cfg_logger=False,
                       separate_figs=True)
            
        if test_config == 41:
            # parameters of the synthetic Boolean MTL problem
            attributes = 12
            disjunct_degree = 6
            n = 50
            task_groups = 5
            noise = 0.0
            data_rnd_seed = 12
            n_learning_sets = 10
            # parameters of the MTL problem tester
            rnd_seed = 51
            results_path_fmt = os.path.join(path_prefix, "results/synthetic_"
                "data/changing_tasks_per_group/bool_func-a{}d{}n{}g{}tg{{}}"
                "nse{}rs{}nls{}-seed{}-complete_test".format(attributes,
                disjunct_degree, n, task_groups, noise, data_rnd_seed,
                n_learning_sets, rnd_seed))
            
            # dynamic parameters of the synthetic Boolean MTL problem
            tasks_per_group_values = [1, 2, 5, 10]
            
            if "run" in mode:
                for tasks_per_group in tasks_per_group_values:
                    # prepare directories and loggers
                    results_path = results_path_fmt.format(tasks_per_group)
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    log_file = os.path.join(results_path, "run-{}.log".\
                                format(time.strftime("%Y%m%d_%H%M%S")))
                    configure_logger(logger, console_level=logging.INFO,
                                     file_name=log_file)
                    log_base_learner_info(logger, base_learners_bool)
                    # generate boolean data with complete test sets
                    funcs_pickle_path = os.path.join(results_path,
                                                     "boolean_funcs.pkl")
                    tasks_data, tasks_complete_test_sets = \
                        synthetic_data.generate_boolean_data_with_complete_test_sets(
                            attributes, disjunct_degree, n, task_groups,
                            tasks_per_group, noise, random_seed=data_rnd_seed,
                            n_learning_sets=n_learning_sets,
                            funcs_pickle_path=funcs_pickle_path)
                    # test the generated MTL problem
                    test_tasks(tasks_data, results_path, base_learners_bool,
                               measures_clas, learners, "pre-prepared_test",
                               rnd_seed=rnd_seed, test=test, unpickle=unpickle,
                               visualize=visualize,
                               preprepared_test_sets=tasks_complete_test_sets,
                               separate_figs=True, cfg_logger=False)
            if "combine" in mode:
                combine_experiment_results(results_path_fmt,
                    tasks_per_group_values,
                    (results_path_fmt.format(tasks_per_group_values) +
                     "-{}-{{}}.pdf".format(error_measure)),
                    n_learning_sets, error_measure=error_measure,
                    title="Avg. results for tasks",
                    xlabel="# of tasks per group")
        
        if test_config == 42:
            # parameters of the synthetic Boolean MTL problem
            attributes = 12
            disjunct_degree = 6
            n = 50
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            n_learning_sets = 10
            # parameters of the MTL problem tester
            rnd_seed = 51
            results_path_fmt = os.path.join(path_prefix, "results/synthetic_"
                "data/changing_task_groups/bool_func-a{}d{}n{}g{{}}tg{}"
                "nse{}rs{}nls{}-seed{}-complete_test".format(attributes,
                disjunct_degree, n, tasks_per_group, noise, data_rnd_seed,
                n_learning_sets, rnd_seed))
            
            # dynamic parameters of the synthetic Boolean MTL problem
            task_groups_values = [1, 2, 5, 10]
            
            if "run" in mode:
                for task_groups in task_groups_values:
                    # prepare directories and loggers
                    results_path = results_path_fmt.format(task_groups)
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    log_file = os.path.join(results_path, "run-{}.log".\
                                format(time.strftime("%Y%m%d_%H%M%S")))
                    configure_logger(logger, console_level=logging.INFO,
                                     file_name=log_file)
                    log_base_learner_info(logger, base_learners_bool)
                    # generate boolean data with complete test sets
                    funcs_pickle_path = os.path.join(results_path,
                                                     "boolean_funcs.pkl")
                    tasks_data, tasks_complete_test_sets = \
                        synthetic_data.generate_boolean_data_with_complete_test_sets(
                            attributes, disjunct_degree, n, task_groups,
                            tasks_per_group, noise, random_seed=data_rnd_seed,
                            n_learning_sets=n_learning_sets,
                            funcs_pickle_path=funcs_pickle_path)
                    # test the generated MTL problem
                    test_tasks(tasks_data, results_path, base_learners_bool,
                               measures_clas, learners, "pre-prepared_test",
                               rnd_seed=rnd_seed, test=test, unpickle=unpickle,
                               visualize=visualize,
                               preprepared_test_sets=tasks_complete_test_sets,
                               separate_figs=True, cfg_logger=False)
            if "combine" in mode:
                combine_experiment_results(results_path_fmt,
                    task_groups_values,
                    (results_path_fmt.format(task_groups_values) + 
                     "-{}-{{}}.pdf".format(error_measure)),
                    n_learning_sets, error_measure=error_measure,
                    title="Avg. results for tasks",
                    xlabel="# of task groups")
    
        if test_config == 43:
            # parameters of the synthetic Boolean MTL problem
            attributes = 12
            disjunct_degree = 6
            task_groups = 5
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            n_learning_sets = 10
            # parameters of the MTL problem tester
            rnd_seed = 51
            results_path_fmt = os.path.join(path_prefix, "results/synthetic_"
                "data/changing_n/bool_func-a{}d{}n{{}}g{}tg{}"
                "nse{}rs{}nls{}-seed{}-complete_test".format(attributes,
                disjunct_degree, task_groups, tasks_per_group, noise,
                data_rnd_seed, n_learning_sets, rnd_seed))
            
            # dynamic parameters of the synthetic Boolean MTL problem
            n_values = [10, 20, 50, 100, 200]
            
            if "run" in mode:
                for n in n_values:
                    # prepare directories and loggers
                    results_path = results_path_fmt.format(n)
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    log_file = os.path.join(results_path, "run-{}.log".\
                                format(time.strftime("%Y%m%d_%H%M%S")))
                    configure_logger(logger, console_level=logging.INFO,
                                     file_name=log_file)
                    log_base_learner_info(logger, base_learners_bool)
                    # generate boolean data with complete test sets
                    funcs_pickle_path = os.path.join(results_path,
                                                     "boolean_funcs.pkl")
                    tasks_data, tasks_complete_test_sets = \
                        synthetic_data.generate_boolean_data_with_complete_test_sets(
                            attributes, disjunct_degree, n, task_groups,
                            tasks_per_group, noise, random_seed=data_rnd_seed,
                            n_learning_sets=n_learning_sets,
                            funcs_pickle_path=funcs_pickle_path)
                    # test the generated MTL problem
                    test_tasks(tasks_data, results_path, base_learners_bool,
                               measures_clas, learners, "pre-prepared_test",
                               rnd_seed=rnd_seed, test=test, unpickle=unpickle,
                               visualize=visualize,
                               preprepared_test_sets=tasks_complete_test_sets,
                               separate_figs=True, cfg_logger=False)
            if "combine" in mode:
                combine_experiment_results(results_path_fmt,
                    n_values,
                    (results_path_fmt.format(n_values) +
                     "-{}-{{}}.pdf".format(error_measure)),
                    n_learning_sets, error_measure=error_measure,
                    title="Avg. results for tasks",
                    xlabel="# of examples")
        
        if test_config == 44:
            # parameters of the synthetic Boolean MTL problem
            attributes = 12
            disjunct_degree = 6
            n = 50
            task_groups = 5
            tasks_per_group = 5
            noise = 0.0
            data_rnd_seed = 12
            n_learning_sets = 10
            # parameters of the MTL problem tester
            rnd_seed = 51
            results_path_fmt = os.path.join(path_prefix, "results/synthetic_"
                "data/changing_noise/bool_func-a{}d{}n{}g{}tg{}"
                "nse{{}}rs{}nls{}-seed{}-complete_test".format(attributes,
                disjunct_degree, n, task_groups, tasks_per_group, data_rnd_seed,
                n_learning_sets, rnd_seed))
            
            # dynamic parameters of the synthetic Boolean MTL problem
            noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            
            if "run" in mode:
                for noise in noise_values:
                    # prepare directories and loggers
                    results_path = results_path_fmt.format(noise)
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    log_file = os.path.join(results_path, "run-{}.log".\
                                format(time.strftime("%Y%m%d_%H%M%S")))
                    configure_logger(logger, console_level=logging.INFO,
                                     file_name=log_file)
                    log_base_learner_info(logger, base_learners_bool)
                    # generate boolean data with complete test sets
                    funcs_pickle_path = os.path.join(results_path,
                                                     "boolean_funcs.pkl")
                    tasks_data, tasks_complete_test_sets = \
                        synthetic_data.generate_boolean_data_with_complete_test_sets(
                            attributes, disjunct_degree, n, task_groups,
                            tasks_per_group, noise, random_seed=data_rnd_seed,
                            n_learning_sets=n_learning_sets,
                            funcs_pickle_path=funcs_pickle_path)
                    # test the generated MTL problem
                    test_tasks(tasks_data, results_path, base_learners_bool,
                               measures_clas, learners, "pre-prepared_test",
                               rnd_seed=rnd_seed, test=test, unpickle=unpickle,
                               visualize=visualize,
                               preprepared_test_sets=tasks_complete_test_sets,
                               separate_figs=True, cfg_logger=False)
            if "combine" in mode:
                combine_experiment_results(results_path_fmt,
                    noise_values,
                    (results_path_fmt.format(noise_values) +
                     "-{}-{{}}.pdf".format(error_measure)),
                    n_learning_sets, error_measure=error_measure,
                    title="Avg. results for tasks", xlabel="% of noise")
