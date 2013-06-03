#
# test.py
# Contains classes and methods for testing and comparing various multi-task
# learning (MTL) algorithms.
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

import bisect, hashlib, logging, os, random, re, time
import cPickle as pickle
from collections import OrderedDict

import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.base import ClassifierMixin, RegressorMixin

from PyMTL import data, stat
from PyMTL.learning import prefiltering, learning
from PyMTL.plotting import BarPlotDesc, LinePlotDesc, plot_multiple, \
    plot_dendrograms
from PyMTL.sklearn_utils import absolute_error, squared_error
from PyMTL.util import logger, configure_logger


def pickle_obj(obj, file_path):
    """Pickle the given object to the given file_path.
    
    Keyword arguments:
    file_path -- string representing the path to the file where to pickle
        the object
    
    """
    with open(file_path, "wb") as pkl_file:
        pickle.dump(obj, pkl_file, pickle.HIGHEST_PROTOCOL)


def unpickle_obj(file_path):
    """Unpickle an object from the given file_path.
    Return the reference to the unpickled object.
    
    Keyword arguments:
    file_path -- string representing the path to the file where the object is
        pickled
    
    """
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)

class Task:
    
    """Contains data pertaining to a particular MTL task and methods for
    extracting and manipulating this data.
    
    """
    
    def __init__(self, id, learn, test):
        """Initialize a Task object. Store the task's id and its learning and
        testing data to private attributes.
        
        Arguments:
        id -- string representing task's id
        learn -- tuple (X, y), where:
            X -- numpy.array representing learning examples
            y -- numpy.array representing learning examples' class values
        test -- a tuple (X, y), where:
            X -- numpy.array representing testing examples
            y -- numpy.array representing testing examples' class values
        
        """
        self.id = id
        self._learn = learn
        self._test = test
        # flatten the target arrays if needed
        # NOTE: This is necessary for some MTL methods.
        if len(self._learn[1].shape) == 2:
            self._learn = self._learn[0], np.ravel(self._learn[1])
        if len(self._test[1].shape) == 2:
            self._test = self._test[0], np.ravel(self._test[1])
    
    def __str__(self):
        """Return a "pretty" representation of the task by indicating its id."""
        return self.id

    def get_learn_data(self):
        """Return the learning data as a tuple of instances and their class
        values.
        
        """
        return self._learn
    
    def get_learn_size(self):
        """Return the number of examples in the learning data. """
        return len(self._learn[0])
    
    def get_test_data(self):
        """Return the testing data as a tuple of instances and their class
        values.
        
        """
        return self._test
    
    def get_test_size(self):
        """Return the number of examples in the testing data. """
        return len(self._test[0])
    
    def get_hash(self):
        """Return a SHA1 hex digest based on task's id and its learning and
        testing data. 
        
        NOTE: The hash built-in is not used since it doesn't work on mutable
        object types such as NumPy arrays.
        
        """
        h = hashlib.sha1()
        h.update(self.id)
        # self._learn and self._test are tuples and can't be put into h.update()
        for l in self._learn:
            h.update(l)
        for t in self._test:
            h.update(t)
        return h.hexdigest()


class CVTask(Task):
    
    """Sub-class of the Task class.
    Contains support for dividing data into folds to perform cross-validation.
    
    """
    
    def __init__(self, id, data):
        """Initialize a CVTask object. Store the task's id and its data to private
        attributes.
        
        Arguments:
        id -- string representing task's id
        data -- sklearn.datasets.Bunch object holding task's data
        
        """
        self.id = id
        self._data = data
        # flatten the target array if needed
        # NOTE: This is necessary for some MTL methods.
        if len(self._data.target.shape) == 2:
            self._data.target = np.ravel(self._data.target)
        self._active_fold = None
    
    def get_data_size(self):
        """Return the number of examples of the task."""
        return self._data.data.shape[0]
    
    def divide_data_into_folds(self, k, rand_seed):
        """Divide the task's data into the given number of folds.
        Store the random indices in the self._cv_folds variable.
        
        Keyword arguments:
        k -- integer representing the number of folds
        rand_seed -- integer representing the seed to use when initializing a
            local random number generator
        
        """
        self._k = k
        self._active_fold = None
        self._cv_folds = list(cross_validation.KFold(self.get_data_size(), k,
            indices=False, shuffle=True, random_state=rand_seed))
    
    def set_active_fold(self, i):
        """Set the active fold to fold i.
        This affects the return values of methods get_learn_data() and
        get_test_data().
        
        Keyword arguments:
        i -- integer representing the fold to activate
        
        """
        if not 0 <= i < self._k:
            raise ValueError("Fold {} doesn't exist!".format(i))
        self._active_fold = i
        learn_ind, test_ind = self._cv_folds[i]
        self._learn = self._data.data[learn_ind], self._data.target[learn_ind]
        self._test = self._data.data[test_ind], self._data.target[test_ind]

    def get_learn_data(self):
        """Return the currently active learn data as a tuple of instances and
        their target values.
        
        """
        if self._active_fold == None:
            raise ValueError("There is no active fold!")
        return self._learn
    
    def get_test_data(self):
        """Return the currently active test data as a tuple of instances and
        their target values.
        
        """
        if self._active_fold == None:
            raise ValueError("There is no active fold!")
        return self._test
    
    def get_hash(self):
        """Return a SHA1 hex digest based on task's id, data and
        cross-validation random indices. 
        
        NOTE: The hash built-in is not used since it doesn't work on mutable
        object types such as NumPy arrays.
        
        """
        h = hashlib.sha1()
        h.update(self.id)
        # self._data is a Bunch object and can't be put into h.update() since it
        # isn't convertible to a buffer
        h.update(self._data.data)
        h.update(self._data.target)
        # self._cv_folds is a list and can't be put into h.update()
        for fold_learn, fold_test in self._cv_folds:
            h.update(fold_learn)
            h.update(fold_test)
        return h.hexdigest()


def _compute_avg_scores(fold_scores):
    """Compute the average scores of the given fold scores.
    Return a four-dimensional dictionary with:
        first key corresponding to the base learner's name,
        second key corresponding to the learner's name,
        third key corresponding to the task's id,
        fourth key corresponding to the scoring measure's name,
        value corresponding to the average value of the scoring measure.
    
    Keyword arguments:
    fold_scores -- five-dimensional dictionary with:
        first key corresponding to the fold number,
        second key corresponding to the base learner's name,
        third key corresponding to the learner's name,
        fourth key corresponding to the task's id,
        fifth key corresponding to the scoring measure's name,
        value corresponding to the scoring measure's value.
    
    """
    avg_scores = dict()
    for bl in fold_scores[0]:
        avg_scores[bl] = dict()
        for l in fold_scores[0][bl]:
            avg_scores[bl][l] = dict()
            for task_id in fold_scores[0][bl][l]:
                avg_scores[bl][l][task_id] = dict()
                for m_name in fold_scores[0][bl][l][task_id]:
                    t_scores = []
                    for i in fold_scores:
                        u_score = fold_scores[i][bl][l][task_id][m_name]
                        if u_score != None:
                            t_scores.append(u_score)
                    # the number of scores for each task is not always the
                    # same since it could happen that in some folds a
                    # scoring measures could not be computed
                    avg_scores[bl][l][task_id][m_name] = (sum(t_scores) /
                                                            len(t_scores))
    return avg_scores


class TestingResults:

    """Contains data of testing a particular base learning method on a
    multi-task learning (MTL) problem.
    
    """
    
    def __init__(self, name, task_hashes, task_sizes, scores, dend_info):
        """Initialize a TestingResults object. Store the given arguments as
        attributes.
        
        Arguments:
        name -- string representing the base learner's name
        task_hashes -- OrderedDictionary with keys corresponding to tasks' ids
            and values to tasks' hashes
        task_sizes -- OrderedDictionary with keys corresponding to tasks' ids
            and values to the number of examples of the task
        scores -- three-dimensional dictionary with:
            first key corresponding to the learner's name,
            second key corresponding to the task's id,
            third key corresponding to the scoring measure's name,
            value corresponding to a list of values of the scoring measure.
        dend_info -- ordered dictionary with keys corresponding to the
            experiment's repetition numbers and values corresponding to lists of
            tuples (one for each merged task) as returned by the
            convert_merg_history_to_scipy_linkage function
        
        """
        self.name = name
        self.task_hashes = task_hashes
        self.task_sizes = task_sizes
        self.scores = scores
        self.dend_info = dend_info


class MTLTester:
    
    """Contains methods for testing various learning algorithms on the given
    multi-task learning (MTL) problem.
    
    """
    
    def __init__(self, tasks_data, seed, repeats, **kwargs):
        """Copy the given tasks' data and number of repetitions as private
        attributes.
        Create a private Random object with the given seed and store it in the
        self._random variable.
        Store the given keyword arguments as a private attribute. The keyword
        arguments are later passed to the _prepare_tasks_data() function.
        
        Arguments:
        tasks_data -- list of Bunch objects, where each Bunch object holds data
            corresponding to a task of the MTL problem 
        seed -- integer to be used as a seed for the private Random object
        repeats -- integer representing how many times the testing experiment
            should be repeated
        
        """
        self._random = random.Random(seed)
        self._repeats = repeats
        self._tasks_data = tasks_data
        self._tasks_data_params = kwargs
        # dictionary that will hold the TestingResults objects, one for each
        # tested base learner
        self._test_res = OrderedDict()
    
    def __str__(self):
        """Return a "pretty" representation of the MTL problem by indicating
        tasks' ids.
        
        """
        return "{} tasks: ".format(len(self._tasks)) + \
            ",".join(sorted(self._tasks.iterkeys()))
    
    def only_keep_k_tasks(self, k):
        """Reduce the size of the MTL problem to k randomly chosen tasks.
        If the MTL problem's size is smaller than k, keep all tasks.
        
        Note: This is useful when the number of tasks is large (> 100), since
        it makes some MTL methods (e.g. ERM) computationally tractable.
        
        Arguments:
        k -- integer representing the number of tasks to keep
        
        """
        new_tasks_data = list()
        for _ in range(min(k, len(self._tasks_data))):
            r = self._random.randrange(0, len(self._tasks_data))
            new_tasks_data.append(self._tasks_data.pop(r))
        logger.info("Kept {} randomly chosen tasks, {} tasks discarded".\
                     format(len(new_tasks_data), len(self._tasks_data)))
        self._tasks_data = new_tasks_data
    
    def get_base_learners(self):
        """Return a tuple with the names of the base learning algorithms that
        have their results stored in the self._test_res TestingResults object.
        
        """
        return tuple(self._test_res.iterkeys())
    
    def get_learners(self):
        """Return a tuple with the names of the learning algorithms that have
        have their results stored in the self._test_res TestingResults object.
        
        """
        rnd_bl = self.get_base_learners()[0]
        return tuple(self._test_res[rnd_bl].scores.iterkeys())
    
    def get_measures(self):
        """Return a tuple with the names of the scoring measures that have
        their results stored in the self._test_res TestingResults object.
        
        """
        rnd_bl = self.get_base_learners()[0]
        rnd_l = self.get_learners()[0]
        rnd_u = tuple(self._test_res[rnd_bl].scores[rnd_l].iterkeys())[0]
        return tuple(self._test_res[rnd_bl].scores[rnd_l][rnd_u].iterkeys())
    
    def _prepare_tasks_data(self, test_prop=0.3):
        """Iterate through the tasks' data stored in self._tasks_data and create
        a new Task object for each task.
        Create a dictionary mapping from tasks' ids to their Task objects and
        store it in the self._tasks variable.
        
        Keyword arguments:
        test_prop -- float (in range 0.0 - 1.0) indicating the proportion of
            data that should be used for testing
        
        """
        self._tasks = OrderedDict()
        for td in self._tasks_data:
            # divide task's data to learn and test sets
            X, y = td.data, td.target
            X_train, X_test, y_train, y_test = cross_validation.\
                train_test_split(X, y, test_size=test_prop,
                                 random_state=self._random.randint(0, 100))
            self._tasks[td.ID] = Task(td.ID, (X_train, y_train),
                                      (X_test, y_test))
    
    def _merge_repetition_scores(self, rpt_scores):
        """Merge the given repetition scores.
        Return a four-dimensional dictionary with:
            first key corresponding to the base learner's name,
            second key corresponding to the learner's name,
            third key corresponding to the task's id (ordered),
            fourth key corresponding to the scoring measure's name,
            value corresponding to a list of values of the scoring measure.
        
        Keyword arguments:
        rpt_scores -- five-dimensional dictionary with:
            first key corresponding to the repetition number (ordered),
            second key corresponding to the base learner's name,
            third key corresponding to the learner's name,
            fourth key corresponding to the task's id,
            fifth key corresponding to the scoring measure's name,
            value corresponding to the scoring measure's value.
        
        """
        mrg_scores = dict()
        for bl in rpt_scores[0]:
            mrg_scores[bl] = dict()
            for l in rpt_scores[0][bl]:
                # prepare an OrderedDict with tasks' ids
                # NOTE: OrderedDict is used to keep the order the tasks so that
                # their results are plotted correctly.
                mrg_scores[bl][l] = OrderedDict([(tid, dict()) for tid in
                                                 self._tasks])
                for task_id in rpt_scores[0][bl][l]:
                    for m_name in rpt_scores[0][bl][l][task_id]:
                        t_scores = []
                        for i in rpt_scores:
                            t_score = rpt_scores[i][bl][l][task_id][m_name]
                            if t_score != None:
                                t_scores.append(t_score)
                        mrg_scores[bl][l][task_id][m_name] = t_scores
        return mrg_scores
    
    def _test_tasks(self, models, measures):
        """Test the given tasks' models on their testing data sets. Compute
        the given scoring measures of the testing results.
        Return a two-dimensional dictionary with the first key corresponding to
        the task's id and the second key corresponding to the measure's name.
        The value corresponds to the score for the given task and scoring
        measure.
        Note: If a particular scoring measure couldn't be computed for a task,
        its value is set to None.
        
        Arguments:
        models -- dictionary mapping from tasks' ids to their models
        measures -- list of strings representing measure's names (currently,
            only CA and AUC are supported)
        
        """
        scores = dict()
        comp_errors = {measure : 0 for measure in measures}
        for tid, task in self._tasks.iteritems():
            scores[tid] = dict()
            X_test, y_test = task.get_test_data()
            y_pred = models[tid].predict(X_test)
            if isinstance(models[tid], ClassifierMixin):
                y_pred_proba = models[tid].predict_proba(X_test)            
            for measure in measures:
                if measure == "AUC":
                    try:
                        # the auc_score function only needs probability
                        # estimates of the positive class
                        score = metrics.auc_score(y_test, y_pred_proba[:, 1])
                    except ValueError as e:
                        if (e.args[0] == 
                            "AUC is defined for binary classification only"):
                            # AUC cannot be computed because all instances
                            # belong to the same class
                            score = None
                            comp_errors[measure] += 1
                        else:
                            raise e
                elif measure == "CA":
                    score = metrics.accuracy_score(y_test, y_pred)
                elif measure == "MAE":
                    score = metrics.mean_absolute_error(y_test, y_pred)
                elif measure == "MSE":
                    score = metrics.mean_squared_error(y_test, y_pred)
                elif measure == "RMSE":
                    score = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                elif measure == "Explained variance":
                    score = metrics.explained_variance_score(y_test, y_pred)
                else:
                    raise ValueError("Unknown scoring measure: {}".\
                                     format(measure))
                scores[tid][measure] = score
        # report the number of errors when computing the scoring measures
        n = len(self._tasks)
        for m_name, m_errors in comp_errors.iteritems():
            if m_errors > 0:
                logger.info("Scoring measure {} could not be computed for {}"
                    " out of {} tasks ({:.1f}%)".format(m_name, m_errors, n,
                    100.*m_errors/n))
        return scores
    
    def _process_repetition_scores(self, rpt_scores, dend_info):
        """Combine the scores of the given repetitions of the experiment and
        store them in TestingResults objects, one for each base learner, along
        with tasks' hashes (used for comparing the results of multiple
        experiments) and dend_info objects (used for plotting dendrograms
        showing merging history of the ERM MTL method).
        Store the created TestingResults objects in self._test_res, which is
        a dictionary with keys corresponding to base learner's names and values
        corresponding to their TestingResults objects.
        
        Arguments:
        rpt_scores -- five-dimensional dictionary with:
            first key corresponding to the repetition number (ordered),
            second key corresponding to the base learner's name,
            third key corresponding to the learner's name,
            fourth key corresponding to the task's id,
            fifth key corresponding to the scoring measure's name,
            value corresponding to the scoring measure's value.
        dend_info -- two-dimensional dictionary with:
            first key corresponding to the base learner's name,
            second key corresponding to the repetition number (ordered)
            value corresponding to a dend_info objects as described in the
            plotting.plot_dendrograms's docstring
        
        """
        # merge results of all repetitions
        scores = self._merge_repetition_scores(rpt_scores)
        # get tasks' hashes and sizes
        task_hashes = OrderedDict()
        task_sizes = OrderedDict()
        for tid, task in self._tasks.iteritems():
            task_hashes[tid] = task.get_hash()
            task_sizes[tid] = task.get_learn_size() + task.get_test_size()
        # store the scores and dend_info objects of each base learner in a
        # separate TestingResults object
        for bl in scores:
            self._test_res[bl] = TestingResults(bl, task_hashes, task_sizes,
                                                scores[bl], dend_info[bl])
    
    def test_tasks(self, learners, base_learners, measures, results_path):
        """Repeat the following experiment self._repeats times:
        Prepare tasks' data with the _prepare_tasks_data() function.
        Test the performance of the given learning algorithms with the given
        base learning algorithms and compute the testing results using the
        given scoring measures.
        Process the obtained repetition scores with the
        _process_repetition_scores() function.
        
        Arguments:
        learners -- ordered dictionary with items of the form (name, learner),
            where name is a string representing the learner's name and
            learner is a MTL method (e.g. ERM, NoMerging, ...) 
        base learners -- ordered dictionary with items of the form (name,
            learner), where name is a string representing the base learner's
            name and learner is a scikit-learn estimator object
        measures -- list of strings representing measure's names (currently,
            only CA and AUC are supported)
        results_path -- string representing the path where to save any extra
            information about the running of this test (currently, only used
            for pickling the results when there is an error in calling the
            learner)
        
        """
        rpt_scores = OrderedDict()
        dend_info = {bl : OrderedDict() for bl in base_learners.iterkeys()}
        for i in range(self._repeats):
            self._prepare_tasks_data(**self._tasks_data_params)
            rpt_scores[i] = {bl : dict() for bl in base_learners.iterkeys()}
            for bl in base_learners:
                for l in learners:
                    start = time.clock()
                    try:
                        R = learners[l](self._tasks, base_learners[bl])
                    except Exception as e:
                        logger.exception("There was an error during repetition:"
                            " {} with base learner: {} and learner: {}.".\
                            format(i, bl, l))
                        if i > 0:
                            logger.info("Saving the results of previous "
                                        "repetitions.")
                            # remove the scores of the last repetition
                            del rpt_scores[i]
                            # process the remaining repetition scores
                            self._process_repetition_scores(rpt_scores,
                                                            dend_info)
                            # pickle them to a file
                            pickle_path_fmt = os.path.join(results_path,
                                                           "bl-{}.pkl")
                            self.pickle_test_results(pickle_path_fmt)
                        # re-raise the original exception
                        import sys
                        exc_info = sys.exc_info()
                        raise exc_info[1], None, exc_info[2]
                    rpt_scores[i][bl][l] = self._test_tasks(R["task_models"],
                                                            measures)
                    end = time.clock()
                    logger.debug("Finished repetition: {}, base learner: {}, "
                        "learner: {} in {:.2f}s".format(i, bl, l, end-start))
                    # store dendrogram info if the results contain it 
                    if "dend_info" in R:
                        dend_info[bl][i] = R["dend_info"]
        self._process_repetition_scores(rpt_scores, dend_info)
    
    def pickle_test_results(self, pickle_path_fmt):
        """Pickle the TestingResults objects in self._test_res to the given
        location.
        
        pickle_path_fmt -- string representing a template for the pickle paths;
            it must contain exactly one pair of braces ({}), where the base
            learner's name will be put
        
        """
        for bl, tr in self._test_res.iteritems():
            pickle_path = pickle_path_fmt.format(bl)
            pickle_obj(tr, pickle_path)
            logger.debug("Successfully pickled the results of base learner: {}"
                         " to file: {}".format(bl, pickle_path))
            
    def find_pickled_test_results(self, pickle_path_fmt):
        """Find previously pickled TestingResults objects in the given location,
        unpickle them and store them in the self._test_res dictionary.
        
        pickle_path_fmt -- string representing a template that was used for
            for the pickle paths; it must contain exactly one pair of braces
            ({}), which were replaced with the base learner's name
        
        """
        dir_name, file_name = os.path.split(pickle_path_fmt)
        # check if pickle_path_fmt has the right format
        match = re.search(r"^(.*){}(.*)$", file_name)
        if not match:
            raise ValueError("The given pickle_path_fmt does not have an "
                "appropriate format.")
        re_template = r"^{}(.+){}$".format(match.group(1), match.group(2)) 
        for file_ in sorted(os.listdir(dir_name)):
            match = re.search(re_template, file_)
            if match:
                bl = match.group(1)
                if bl not in self._test_res:
                    file_path = os.path.join(dir_name, file_)
                    tr = unpickle_obj(file_path)
                    if not isinstance(tr, TestingResults):
                        raise TypeError("Object loaded from file: {} is not "
                            "of type TestingResults.".format(file_path))
                    self._test_res[bl] = tr
                    logger.debug("Successfully unpickled the results of base"
                        " learner: {} from file: {}".format(bl, file_path))
                else:
                    logger.info("Results of base learner: {} are already "
                        "loaded".format(bl))
        
    def check_test_results_compatible(self):
        """Check if all TestingResults objects in the self._test_res dictionary
        had the same tasks, tasks' data tables and cross-validation indices.
        In addition, check if all TestingResults objects had the same learning
        algorithms and scoring measures.
        
        """
        bls = self.get_base_learners()
        if len(bls) <= 1:
            return True
        # select the first base learner as the reference base learner
        ref = bls[0]
        test_res_ref = self._test_res[ref]
        for bl in bls[1:]:
            test_res_bl = self._test_res[bl]
            # check if tasks' ids and hashes match for all base learning
            # algorithms
            if len(test_res_bl.task_hashes) != len(test_res_ref.task_hashes):
                return False
            for id, h in test_res_ref.task_hashes.iteritems():
                if test_res_bl.task_hashes[id] != h:
                    return False
            # check if learning algorithms match for all base learning
            # algorithms
            if len(test_res_bl.scores) != len(test_res_ref.scores):
                return False
            for l in test_res_ref.scores:
                if l not in test_res_bl.scores:
                    return False
            # check if scoring measures match for all combinations of base
            # learning algorithms and learning algorithms
            for l in test_res_ref.scores:
                rnd_id = tuple(test_res_ref.scores[l].iterkeys())[0]
                set_ref = set(test_res_ref.scores[l][rnd_id].iterkeys())
                set_bl = set(test_res_bl.scores[l][rnd_id].iterkeys())
                if set_ref != set_bl:
                    return False
        return True
    
    def contains_test_results(self):
        """Return True if the MTLTester contains any testing results."""
        return len(self._test_res) > 0
    
    def _compute_task_stats(self, base_learner, learner, measure):
        """Compute the statistics (average, std. deviation and 95% confidence
        interval) of the performance of the given base learner and learner with
        the given measure for each task.
        Return a triple (avgs, stds, ci95s), where:
            avgs -- list of averages, one for each task
            stds -- list of standard deviations, one for each task
            ci95s -- list of 95% confidence intervals for the means, one for
                each task
        
        """
        # prepare lists that will store the results
        avgs = []
        stds = []
        ci95s = []
        # get the scores for the given base learner and learner
        bl_l_scores = self._test_res[base_learner].scores[learner]
        for tid in bl_l_scores:
            # get the scores of the current task for the given scoring measure
            scores = bl_l_scores[tid][measure]
            avgs.append(stat.mean(scores))
            stds.append(stat.unbiased_std(scores))
            ci95s.append(stat.ci95(scores))
        return avgs, stds, ci95s
    
    def visualize_results(self, base_learners, learners, measures,
                              results_path, colors, error_bars):
        """Visualize the results of the given learning algorithms with the given
        base learning algorithms and the given scoring measures on the MTL
        problem.
        Compute the averages, std. deviations and 95% conf. intervals on bins
        of tasks for all combinations of learners, base learners and scoring
        measures.
        Draw a big plot displaying the averages and std. deviations for each
        scoring measure. Each big plot has one subplot for each base learner.
        Each subplot shows the comparison between different learning algorithms.
        The same big plots are drawn for averages and 95% conf. intervals.
        Save the drawn plots to the given results' path.
        
        Arguments:
        base_learners -- list of strings representing the names of base learners
        learners -- list of strings representing the names of learners
        measures -- list of strings representing names of the scoring measures
        results_path -- string representing the path where to save the generated
            plots
        colors -- dictionary mapping from learners' names to the colors that
            should represent them in the plots
        error_bars -- boolean indicating whether to plot the error bars when
            visualizing the results
        
        """
        for m in measures:
            # default y-axis limits
            ylim_bottom = 0
            ylim_top = None
            if m in ["CA", "AUC"]:
                ylim_top = 1
            # x points and labels
            x_labels = list(self._test_res[base_learners[0]].\
                            scores[learners[0]].keys())
            x_points = np.arange(len(x_labels))
            # plot descriptions for averages and std. deviations
            plot_desc_sd = OrderedDict()
            # plot descriptions for averages and 95% conf. intervals
            plot_desc_ci95 = OrderedDict()
            for bl in base_learners:
                plot_desc_sd[bl] = []
                plot_desc_ci95[bl] = []
                for l in learners:
                    avgs, stds, ci95s = self._compute_task_stats(bl, l, m)
                    plot_desc_sd[bl].append(LinePlotDesc(x_points,
                        avgs, stds, l, color=colors[l], ecolor=colors[l]))
                    plot_desc_ci95[bl].append(LinePlotDesc(x_points,
                        avgs, ci95s, l, color=colors[l], ecolor=colors[l]))
            plot_multiple(plot_desc_sd,
                os.path.join(results_path, "{}-avg-SD.pdf".format(m)),
                title="Avg. results for tasks (error bars show std. dev.)",
                subplot_title_fmt="Learner: {}",
                xlabel="Task name",
                ylabel=m,
                x_tick_points=x_points,
                x_tick_labels=x_labels,
                ylim_bottom=ylim_bottom, ylim_top=ylim_top,
                error_bars=error_bars)
            plot_multiple(plot_desc_ci95,
                os.path.join(results_path, "{}-avg-CI.pdf".format(m)),
                title="Avg. results for tasks (error bars show 95% conf. "
                    "intervals)",
                subplot_title_fmt="Learner: {}",
                xlabel="Task name",
                ylabel=m,
                x_tick_points=x_points,
                x_tick_labels=x_labels,
                ylim_bottom=ylim_bottom, ylim_top=ylim_top,
                error_bars=error_bars)
    
    def visualize_dendrograms(self, base_learners, results_path):
        """Visualize the dendrograms showing merging history of the ERM MTL
        method with the given base learning algorithms.
        Save the draws plots (one for each repetition of the experiment) to
        the given results' path.
        
        Arguments:
        base_learners -- list of strings representing the names of base learners
        results_path -- string representing the path where to save the generated
            dendrograms
        
        """
        for bl in base_learners:
            if len(self._test_res[bl].dend_info) > 0:
                for i, dend_info in self._test_res[bl].dend_info.iteritems():
                    save_path = os.path.join(results_path, "dend-{}-repeat{}."
                                             "pdf".format(bl, i))
                    plot_dendrograms(dend_info, save_path, title="Merging "
                                     "history of ERM with base learner {} "
                                     "(repetition {})".format(bl, i))
    
    def _compute_overall_stats(self, base_learner, learner, measure,
                               weighting="all_equal"):
        """Compute the overall results for the given learning algorithm,
        base learning algorithm and scoring measure.
        First, compute the weighted average of the scoring measure over all
        tasks. Return the average, std. deviation and 95% conf.
        interval of these values over all repetitions of the experiment.
        
        Arguments:
        base_learner -- string representing the name of the base learner
        learner -- strings representing the name of the learner
        measures -- string representing the name of the scoring measure
        
        Keyword arguments:
        weighting -- string representing the weighting method to use when
            computing the average of the scoring measure over all tasks;
            currently, two options are implemented:
            - all_equal -- all tasks have equal weight
            - task_sizes -- tasks' weights correspond to their sizes
        
        """
        # get the scores for the given base learner and learner
        bl_l_scores = self._test_res[base_learner].scores[learner]
        # get the task sizes
        task_sizes = self._test_res[base_learner].task_sizes
        # extract the number of repetitions of the experiment
        rep = len(bl_l_scores[bl_l_scores.keys()[0]][measure])
        # create a matrix, where element (i, j) represents the score of task j
        # at repetition i
        scores = np.zeros((rep, len(task_sizes)))
        for j, tid in enumerate(task_sizes):
            # get the scores of the current task for the given scoring measure
            scores[:, j] = bl_l_scores[tid][measure]
        # compute the weighted average scores across all tasks
        if weighting == "all_equal":
            weights = np.ones(len(task_sizes))
        elif weighting == "task_sizes":
            weights = np.array(list(task_sizes.values()))
        else:
            raise ValueError("Unknown weighting method: {}".format(weighting))
        avg_scores = np.dot(scores, weights) / np.sum(weights)
        # compute the average of the above average scores across all repetitions
        # of the experiment
        return (stat.mean(avg_scores), stat.unbiased_std(avg_scores),
                stat.ci95(avg_scores))
    
    def compute_overall_results(self, base_learners, learners, measures,
            results_path, weighting="all_equal", error_margin="std"):
        """Compute the overall results for the given learning algorithms with
        the given base learning algorithms and the given scoring measures on
        the MTL problem.
        First, compute the weighted average of the scoring measure over all
        tasks. Then compute the averages, std. deviations and 95% conf.
        intervals of these values over all repetitions of the experiment.
        Write the results to the given results' path.
        
        Arguments:
        base_learners -- list of strings representing the names of base learners
        learners -- list of strings representing the names of learners
        measures -- list of strings representing names of the scoring measures
        results_path -- string representing the path where to write the computed
            results
        
        Keyword arguments:
        weighting -- string representing the weighting method to use when
            computing the average of the scoring measure over all tasks;
            currently, two options are implemented:
            - all_equal -- all tasks have equal weight
            - task_sizes -- tasks' weights correspond to their sizes
        error_margin -- string representing the measure of the margin of error;
            currently, two options are implemented:
            - std -- margin of error is std. deviation
            - ci95 -- margin of error is 95% conf. interval of the mean
        
        """
        offset = "  "
        with open(os.path.join(results_path, "overall_results.txt"), 'w') as r:
            for m in measures:
                s = "Results for {} (weighting method: {}, error margin " \
                    "measure: {})".format(m, weighting, error_margin)
                r.write(s + "\n")
                r.write("-"*len(s) + "\n")
                for bl in base_learners:
                    r.write(offset + "- Base learner: {}".format(bl) + "\n")
                    for l in learners:
                        avg, std, ci95 = self._compute_overall_stats(bl, l, m,
                                                weighting=weighting)
                        if error_margin == "std":
                            em = std
                        elif error_margin == "ci95":
                            em = ci95
                        else:
                            raise ValueError("Unknown error margin measure: "
                                             "{}".format(error_margin))
                        r.write(3*offset + "* {:<20}{:.2f} +/- {:.2f}".\
                                format(l, avg, em) + "\n")
                r.write("\n")

class SubtasksMTLTester(MTLTester):
    
    """Sub-class of the MTLTester which implements dividing tasks into sub-tasks
    by dividing the tasks' learning set into subsets.
    
    """
    
    def _prepare_tasks_data(self, test_prop=0.3, subtasks_split=(3, 5)):
        """Iterate through the tasks' data stored in self._tasks_data and divide
        them to learn and test sets.
        Create a random number of sub-tasks by dividing the tasks' learning set
        to subsets.
        Create a dictionary mapping from tasks' ids to their Task objects and
        store it in the self._tasks variable.
        
        Keyword arguments:
        test_prop -- float (in range 0.0 - 1.0) indicating the proportion of
            data that should be used for testing
        subtasks_split -- tuple of the form (a, b), where a and b indicate the
            minimal and maximal number (respectively) of sub-tasks the task is
            divided into
        
        """
        self._tasks = OrderedDict()
        # list of original tasks' ids
        # NOTE: This is used later to order the tasks when plotting the results.
        self._orig_task_ids = []
        for td in self._tasks_data:
            self._orig_task_ids.append(td.ID)
            # divide task's data to learn and test sets
            X, y = td.data, td.target
            X_train, X_test, y_train, y_test = cross_validation.\
                train_test_split(X, y, test_size=test_prop,
                                 random_state=self._random.randint(0, 100))
            # create a random number of subtasks by dividing the task's learn
            # set to an appropriate number of subsets
            n_subtasks = self._random.choice(range(subtasks_split[0],
                                                   subtasks_split[1] + 1))
            for i, (_, test_m) in enumerate(cross_validation.KFold(len(y_train),
                n_folds=n_subtasks, indices=False, shuffle=True,
                random_state=self._random.randint(0, 100))):
                tid = "{} (part {})".format(td.ID, i + 1)
                learn = X_train[test_m], y_train[test_m]
                test = X_test, y_test
                self._tasks[tid] = Task(tid, learn, test)
            logger.debug("Splitted task '{}' into {} sub-tasks.".format(td.ID,
                                                                    n_subtasks))
    
    def _merge_repetition_scores(self, rpt_scores):
        """Merge the given repetition scores. The scores of all sub-tasks should
        be merged into a single list.
        Return a four-dimensional dictionary with:
            first key corresponding to the base learner's name,
            second key corresponding to the learner's name,
            third key corresponding to the (unsplitted) task's id,
            fourth key corresponding to the scoring measure's name,
            value corresponding to a list of values of the scoring measure.
        
        Keyword arguments:
        rpt_scores -- five-dimensional dictionary with:
            first key corresponding to the repetition number,
            second key corresponding to the base learner's name,
            third key corresponding to the learner's name,
            fourth key corresponding to the (splitted) task's id,
            fifth key corresponding to the scoring measure's name,
            value corresponding to the scoring measure's value.
        
        """
        mrg_scores = dict()
        for bl in rpt_scores[0]:
            mrg_scores[bl] = dict()
            for l in rpt_scores[0][bl]:
                # prepare an OrderedDict with original tasks' ids
                # NOTE: OrderedDict is used to keep the order the tasks so that
                # their results are plotted correctly.
                mrg_scores[bl][l] = OrderedDict([(tid, dict()) for tid in
                                                 self._orig_task_ids])
                for rpt in rpt_scores:
                    for task_id in rpt_scores[rpt][bl][l]:
                        # extract the original task's id from the task's id
                        match = re.search(r"^[^(]+", task_id)
                        if match:
                            orig_task_id = match.group().rstrip()
                        else:
                            raise ValueError("Could not extract the original "
                                    "task's id from '{}'".format(task_id))
                        if orig_task_id not in mrg_scores[bl][l]:
                            raise ValueError("Original task's id: {} not "\
                                             "found.".format(orig_task_id))
                        for m_name in rpt_scores[rpt][bl][l][task_id]:
                            if m_name not in mrg_scores[bl][l][orig_task_id]:
                                mrg_scores[bl][l][orig_task_id][m_name] = []
                            t_score = rpt_scores[rpt][bl][l][task_id][m_name]
                            if t_score != None:
                                mrg_scores[bl][l][orig_task_id][m_name].\
                                    append(t_score)
        return mrg_scores


class CVMTLTester(MTLTester):
    
    """Contains methods for testing various learning algorithms on the given
    multi-task learning (MTL) problem.
    
    """
    
    def __init__(self, tasks_data, seed):
        """Iterate through the given tasks data and create a new CVTask object for
        each task.
        Create a dictionary mapping from tasks' ids to their CVTask objects and
        store it in the self._tasks variable.
        Create a private Random object with the given seed and store it in the
        self._random variable.
        
        Arguments:
        tasks_data -- list of Bunch objects, where each Bunch object holds data
            corresponding to a task of the MTL problem 
        seed -- integer to be used as a seed for the private Random object
        
        """
        self._tasks = OrderedDict()
        for td in tasks_data:
            self._tasks[td.ID] = CVTask(td.ID, td)
        self._random = random.Random(seed)
        # dictionary that will hold the TestingResults objects, one for each
        # tested base learner
        self._test_res = OrderedDict()
    
    def __str__(self):
        """Return a "pretty" representation of the MTL problem by indicating
        tasks' ids.
        
        """
        return "{} tasks: ".format(len(self._tasks)) + \
            ",".join(sorted(self._tasks.iterkeys()))
    
    def only_keep_k_tasks(self, k):
        """Reduce the size of the MTL problem to k randomly chosen tasks.
        If the MTL problem's size is smaller than k, keep all tasks.
        
        Note: This is useful when the number of tasks is large (> 100), since
        it makes some MTL methods (e.g. ERM) computationally tractable.
        
        Arguments:
        k -- integer representing the number of tasks to keep
        
        """
        new_tasks = OrderedDict()
        for _ in range(min(k, len(self._tasks))):
            tid = self._random.choice(self._tasks.keys())
            new_tasks[tid] = self._tasks[tid]
            del self._tasks[tid]
        logger.info("Kept {} randomly chosen tasks, {} tasks discarded".\
                     format(len(new_tasks), len(self._tasks)))
        self._tasks = new_tasks
    
    def get_base_learners(self):
        """Return a tuple with the names of the base learning algorithms that
        have their results stored in the self._test_res TestingResults object.
        
        """
        return tuple(self._test_res.iterkeys())
    
    def get_learners(self):
        """Return a tuple with the names of the learning algorithms that have
        have their results stored in the self._test_res TestingResults object.
        
        """
        rnd_bl = self.get_base_learners()[0]
        return tuple(self._test_res[rnd_bl].avg_scores.iterkeys())
    
    def get_measures(self):
        """Return a tuple with the names of the scoring measures that have
        their results stored in the self._test_res TestingResults object.
        
        """
        rnd_bl = self.get_base_learners()[0]
        rnd_l = self.get_learners()[0]
        rnd_u = tuple(self._test_res[rnd_bl].avg_scores[rnd_l].iterkeys())[0]
        return tuple(self._test_res[rnd_bl].avg_scores[rnd_l][rnd_u].iterkeys())
    
    def _test_tasks(self, models, measures):
        """Test the given tasks' models on their testing data sets. Compute
        the given scoring measures of the testing results.
        Return a two-dimensional dictionary with the first key corresponding to
        the task's id and the second key corresponding to the measure's name.
        The value corresponds to the score for the given task and scoring
        measure.
        Note: If a particular scoring measure couldn't be computed for a task,
        its value is set to None.
        
        Arguments:
        models -- dictionary mapping from tasks' ids to their models
        measures -- list of strings representing measure's names (currently,
            only CA and AUC are supported)
        
        """
        scores = dict()
        comp_errors = {measure : 0 for measure in measures}
        for tid, task in self._tasks.iteritems():
            X_test, y_test = task.get_test_data()
            y_pred = models[tid].predict(X_test)
            y_pred_proba = models[tid].predict_proba(X_test)
            scores[tid] = dict()
            for measure in measures:
                if measure == "AUC":
                    try:
                        # the auc_score function only needs probability
                        # estimates of the positive class
                        score = metrics.auc_score(y_test, y_pred_proba[:, 1])
                    except ValueError as e:
                        if (e.args[0] == 
                            "AUC is defined for binary classification only"):
                            # AUC cannot be computed because all instances
                            # belong to the same class
                            score = None
                            comp_errors[measure] += 1
                        else:
                            raise e
                elif measure == "CA":
                    score = metrics.accuracy_score(y_test, y_pred)
                else:
                    raise ValueError("Unknown scoring measure: {}".\
                                     format(measure))
                scores[tid][measure] = score
        # report the number of errors when computing the scoring measures
        n = len(self._tasks)
        for m_name, m_errors in comp_errors.iteritems():
            if m_errors > 0:
                logger.info("Scoring measure {} could not be computed for {}"
                    " out of {} tasks ({:.1f}%)".format(m_name, m_errors, n,
                    100.*m_errors/n))
        return scores
    
    def test_tasks(self, learners, base_learners, measures, results_path):
        """Divide all tasks' data into folds and perform the tests on each fold.
        Test the performance of the given learning algorithms with the given
        base learning algorithms and compute the testing results using the
        given scoring measures.
        Compute the average scores over all folds and store them in
        TestingResults objects, one for each base learner, along with tasks'
        hashes (used for comparing the results of multiple experiments).
        Store the created TestingResults objects in self._test_res, which is
        a dictionary with keys corresponding to base learner's names and values
        corresponding to their TestingResults objects. 
        
        Arguments:
        learners -- ordered dictionary with items of the form (name, learner),
            where name is a string representing the learner's name and
            learner is a merging learning algorithm (e.g. ERM, NoMerging, ...) 
        base learners -- ordered dictionary with items of the form (name,
            learner), where name is a string representing the base learner's
            name and learner is a scikit-learn estimator object
        measures -- list of strings representing measure's names (currently,
            only CA and AUC are supported)
        results_path -- string representing the path where to save any extra
            information about the running of this test (currently, just ERM's
            dendrograms)
        
        """
        # divide tasks' data into folds
        folds = 5
        for task in self._tasks.itervalues():
            task.divide_data_into_folds(folds, self._random.randint(0, 100))
        # perform learning and testing for each fold
        fold_scores = OrderedDict()
        for i in range(folds):
            for task in self._tasks.itervalues():
                task.set_active_fold(i)
            fold_scores[i] = {bl : dict() for bl in base_learners.iterkeys()}
            for bl in base_learners:
                for l in learners:
                    start = time.clock()
                    R = learners[l](self._tasks, base_learners[bl])
                    fold_scores[i][bl][l] = self._test_tasks(R["task_models"],
                                                             measures)
                    end = time.clock()
                    logger.debug("Finished fold: {}, base learner: {}, "
                        "learner: {} in {:.2f}s".format(i, bl, l, end-start))
                    # plot a dendrogram showing merging history if the results
                    # contain dendrogram info 
                    if "dend_info" in R:
                        plot_dendrograms(R["dend_info"], os.path.join(
                            results_path, "dend-{}-fold{}.png".format(bl, i)),
                            title="Merging history of ERM with base learner {}"
                            " (fold {})".format(bl, i))
        # compute the average measure scores over all folds
        avg_scores = _compute_avg_scores(fold_scores)
        # get tasks' hashes
        task_hashes = OrderedDict()
        for tid, task in self._tasks.iteritems():
            task_hashes[tid] = task.get_hash()
        # store the average scores of each base learner in a separate
        # TestingResults object
        for bl in avg_scores:
            self._test_res[bl] = TestingResults(bl, task_hashes, avg_scores[bl])
    
    def pickle_test_results(self, pickle_path_fmt):
        """Pickle the TestingResults objects in self._test_res to the given
        location.
        
        pickle_path_fmt -- string representing a template for the pickle paths;
            it must contain exactly one pair of braces ({}), where the base
            learner's name will be put
        
        """
        for bl, tr in self._test_res.iteritems():
            pickle_path = pickle_path_fmt.format(bl)
            pickle_obj(tr, pickle_path)
            logger.debug("Successfully pickled the results of base learner: {}"
                         " to file: {}".format(bl, pickle_path))
            
    def find_pickled_test_results(self, pickle_path_fmt):
        """Find previously pickled TestingResults objects in the given location,
        unpickle them and store them in the self._test_res dictionary.
        
        pickle_path_fmt -- string representing a template that was used for
            for the pickle paths; it must contain exactly one pair of braces
            ({}), which were replaced with the base learner's name
        
        """
        dir_name, file_name = os.path.split(pickle_path_fmt)
        # check if pickle_path_fmt has the right format
        match = re.search(r"^(.*){}(.*)$", file_name)
        if not match:
            raise ValueError("The given pickle_path_fmt does not have an "
                "appropriate format.")
        re_template = r"^{}(.+){}$".format(match.group(1), match.group(2)) 
        for file_ in sorted(os.listdir(dir_name)):
            match = re.search(re_template, file_)
            if match:
                bl = match.group(1)
                if bl not in self._test_res:
                    file_path = os.path.join(dir_name, file_)
                    tr = unpickle_obj(file_path)
                    if not isinstance(tr, TestingResults):
                        raise TypeError("Object loaded from file: {} is not "
                            "of type TestingResults.".format(file_path))
                    self._test_res[bl] = tr
                    logger.debug("Successfully unpickled the results of base"
                        " learner: {} from file: {}".format(bl, file_path))
                else:
                    logger.info("Results of base learner: {} are already "
                        "loaded".format(bl))
        
    def check_test_results_compatible(self):
        """Check if all TestingResults objects in the self._test_res dictionary
        had the same tasks, tasks' data tables and cross-validation indices.
        In addition, check if all TestingResults objects had the same learning
        algorithms and scoring measures.
        
        """
        bls = self.get_base_learners()
        if len(bls) <= 1:
            return True
        # select the first base learner as the reference base learner
        ref = bls[0]
        test_res_ref = self._test_res[ref]
        for bl in bls[1:]:
            test_res_bl = self._test_res[bl]
            # check if tasks' ids and hashes match for all base learning
            # algorithms
            if len(test_res_bl.task_hashes) != len(test_res_ref.task_hashes):
                return False
            for id, h in test_res_ref.task_hashes.iteritems():
                if test_res_bl.task_hashes[id] != h:
                    return False
            # check if learning algorithms match for all base learning
            # algorithms
            if len(test_res_bl.avg_scores) != len(test_res_ref.avg_scores):
                return False
            for l in test_res_ref.avg_scores:
                if l not in test_res_bl.avg_scores:
                    return False
            # check if scoring measures match for all combinations of base
            # learning algorithms and learning algorithms
            for l in test_res_ref.avg_scores:
                rnd_id = tuple(test_res_ref.avg_scores[l].iterkeys())[0]
                set_ref = set(test_res_ref.avg_scores[l][rnd_id].iterkeys())
                set_bl = set(test_res_bl.avg_scores[l][rnd_id].iterkeys())
                if set_ref != set_bl:
                    return False
        return True
    
    def _compute_task_stats(self, base_learner, learner, measure):
        """Compute the statistics (average, std. deviation and 95% confidence
        interval) of the performance of the given base learner and learner with
        the given measure for each task in self._tasks.
        Return a triple (avgs, stds, ci95s), where:
            avgs -- list of averages, one for each task
            stds -- list of standard deviations, one for each task
            ci95s -- list of 95% confidence intervals for the means, one for
                each task
        
        """
        # prepare lists that will store the results
        avgs = []
        stds = []
        ci95s = []
        for tid, t in self._tasks.iteritems():
            # get the scores of the current task for the given base learner,
            # learner and scoring measure
            bl_avg_scores = self._test_res[base_learner].avg_scores
            scores = [bl_avg_scores[learner][tid][measure]]
            avgs.append(stat.mean(scores))
            stds.append(stat.unbiased_std(scores))
            ci95s.append(stat.ci95(scores))
        return avgs, stds, ci95s
    
    def visualize_results(self, base_learners, learners, measures,
                              results_path, colors):
        """Visualize the results of the given learning algorithms with the given
        base learning algorithms and the given scoring measures on the MTL
        problem.
        Compute the averages, std. deviations and 95% conf. intervals on bins
        of tasks for all combinations of learners, base learners and scoring
        measures.
        Draw a big plot displaying the averages and std. deviations for each
        scoring measure. Each big plot has one subplot for each base learner.
        Each subplot shows the comparison between different learning algorithms.
        The same big plots are drawn for averages and 95% conf. intervals.
        Save the drawn plots to the files with the given path prefix.
        
        Arguments:
        base_learners -- list of strings representing the names of base learners
        learners -- list of strings representing the names of learners
        measures -- list of strings representing names of the scoring measures
        results_path -- string representing the path where to save the generated
            plots
        colors -- dictionary mapping from learners' names to the colors that
            should represent them in the plots
        
        """
        for m in measures:
            # plot descriptions for averages and std. deviations
            plot_desc_sd = OrderedDict()
            # plot descriptions for averages and 95% conf. intervals
            plot_desc_ci95 = OrderedDict()
            for bl in base_learners:
                plot_desc_sd[bl] = []
                plot_desc_ci95[bl] = []
                for l in learners:
                    avgs, stds, ci95s = self._compute_task_stats(bl, l, m)
                    plot_desc_sd[bl].append(LinePlotDesc(np.arange(len(avgs)),
                        avgs, stds, l, color=colors[l], ecolor=colors[l]))
                    plot_desc_ci95[bl].append(LinePlotDesc(np.arange(len(avgs)),
                        avgs, ci95s, l, color=colors[l], ecolor=colors[l]))
            plot_multiple(plot_desc_sd,
                os.path.join(results_path, "{}-avg-SD.pdf".format(m)),
                title="Avg. results for tasks (error bars show std. dev.)",
                subplot_title_fmt="Learner: {}",
                xlabel="Number of instances",
                ylabel=m)
            plot_multiple(plot_desc_ci95,
                os.path.join(results_path, "{}-avg-CI.pdf".format(m)),
                title="Avg. results for tasks (error bars show 95% conf. "
                    "intervals)",
                subplot_title_fmt="Learner: {}",
                xlabel="Number of instances",
                ylabel=m)

def test_tasks(tasks_data, results_path, base_learners,
               measures, learners, tester_type, rnd_seed=50,
               test=True, unpickle=False, visualize=True,
               test_prop=0.3, subtasks_split=(3, 5), cv_folds=5,
               repeats=1, keep=0, weighting="all_equal", error_margin="std",
               error_bars=True, cfg_logger=True):
    """Test the given tasks' data corresponding to a MTL problem according to
    the given parameters and save the results where indicated.
    
    Arguments:
    tasks_data -- list of Bunch objects that hold tasks' data
    results_path -- string representing the path where to store the results
        (if it doesn't exist, it will be created)
    base_learners -- ordered dictionary with items of the form (name, learner),
        where name is a string representing the base learner's name and learner
        is a scikit-learn estimator object
    measures -- list of strings representing measure's names
    learners -- ordered dictionary with items of the form (name, learner),
        where name is a string representing the learner's name and
        learner is a merging learning algorithm (e.g. ERM, NoMerging, ...)
    tester_type -- string indicating which (sub)class of MTLTester to use;
        currently, only "train_test_split", "subtasks_split" and "cv" are
        supported
    
    Keyword arguments:
    rnd_seed -- integer indicating the random seed to be used for the MTLTester
        object
    test -- boolean indicating whether to perform tests on the MTL problem (with
        the given base_learners, measures and learners)
    unpickle -- boolean indicating whether to search for previously computed
        testing results and including them in the MTL problem
    visualize -- boolean indicating whether to visualize the results of the
        current tasks (for each combination of base learners, measures and
        learners in the MTL problem)
    test_prop -- parameter for MTLTester and SubtasksMTLTester
    subtasks_split -- parameter for the SubtasksMTLTester
    cv_folds -- integer indicating how many folds to use with the CVMTLTester
    repeats -- integer indicating how many times the MTLTester should repeat the
        experiment
    keep -- integer indicating the number of random tasks of the MTL problem to
        keep; if 0 (Default), then all tasks are kept
    weighting -- string indicating the type of weighting to use when computing
        the overall results
    error_margin -- string indicating which measure to use for error margins
        when computing the overall results
    error_bars -- boolean indicating whether to plot the error bars when
        visualizing the results
    cfg_logger -- boolean indicating whether to re-configure the global logger
        object
    
    """
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    pickle_path_fmt = os.path.join(results_path, "bl-{}.pkl")
    if cfg_logger:
        log_file = os.path.join(results_path,
                            "run-{}.log".format(time.strftime("%Y%m%d_%H%M%S")))
        configure_logger(logger, console_level=logging.INFO, file_name=log_file)
    # create a MTL tester with tasks' data
    if tester_type == "train_test_split":
        mtlt = MTLTester(tasks_data, rnd_seed, test_prop=test_prop,
                         repeats=repeats)
    elif tester_type == "subtasks_split":
        mtlt = SubtasksMTLTester(tasks_data, rnd_seed, test_prop=test_prop,
                                 subtasks_split=subtasks_split, repeats=repeats)
    elif tester_type == "cv":
        mtlt = CVMTLTester(tasks_data, rnd_seed, cv_folds=cv_folds)
    else:
        raise ValueError("Unknown MTL tester type: '{}'".format(tester_type))
    # select a random subset of tasks if keep > 0
    if keep > 0:
        mtlt.only_keep_k_tasks(keep)
    # test all combinations of learners and base learners (compute the testing
    # results with the defined measures) and save the results if test == True
    if test:
        mtlt.test_tasks(learners, base_learners, measures, results_path)
        mtlt.pickle_test_results(pickle_path_fmt)
    # find previously computed testing results and check if they were computed
    # using the same data tables and cross-validation indices if
    # unpickle == True
    if unpickle:
        mtlt.find_pickled_test_results(pickle_path_fmt)
        if not mtlt.check_test_results_compatible():
            raise ValueError("Test results for different base learners are not "
                             "compatible.")
    # visualize the results of the current tasks for each combination of base
    # learners, learners and measures that are in the MTL problem; in addition,
    # visualize the dendrograms showing merging history of ERM
    if visualize:
        if not mtlt.contains_test_results():
            raise ValueError("The MTLTester object doesn't contain any testing"
                             " results.")
        bls = mtlt.get_base_learners()
        ls = mtlt.get_learners()
        ms = mtlt.get_measures()
        mtlt.visualize_results(bls, ls, ms, results_path,
            {"NoMerging": "blue", "MergeAll": "green", "ERM": "red"},
            error_bars)
        mtlt.visualize_dendrograms(bls, results_path)
        mtlt.compute_overall_results(bls, ls, ms, results_path,
                weighting=weighting, error_margin=error_margin)

if __name__ == "__main__":
    # boolean indicating which testing configuration to use:
    # 1 -- USPS digits data (repeats=10)
    # 2 -- USPS digits data (repeats=10, subtasks=(3, 5))
    # 3 -- USPS digits data (repeats=10, subtasks=(5, 10))
    # 4 -- MNIST digits data (repeats=10)
    # 5 -- MNIST digits data (repeats=10, subtasks=(3, 5))
    # 6 -- MNIST digits data (repeats=10, subtasks=(5, 10))
    # 7 -- School data
    # 8 -- School data (train-test split is 60-40 instead of 75-25)
    # 9 -- School data (only a subset of tasks)
    # 10 -- Computer survey data
    test_config = 10
    
    # boolean indicating whether to perform the tests on the MTL problem
    test = True
    # boolean indicating whether to find previously computed testing results
    # and unpickling them
    unpickle = False
    # boolean indicating whether to visualize the results of the MTL problem
    visualize = True
    
    # find out the current file's location so it can be used to compute the
    # location of other files/directories
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    
    # base learners for classification problems
    base_learners_clas = OrderedDict()
    from sklearn.linear_model import LogisticRegression
#    from sklearn.pipeline import Pipeline
#    from sklearn_utils import MeanImputer
#    clf = Pipeline([("imputer", MeanImputer()),
#                    ("log_reg", LogisticRegression())])
    clf = LogisticRegression()
    base_learners_clas["log_reg"] = clf
#    from sklearn.dummy import DummyClassifier
#    from sklearn.pipeline import Pipeline
#    from sklearn_utils import MeanImputer
#    clf = Pipeline([("imputer", MeanImputer()),
#                    ("majority", DummyClassifier(strategy="most_frequent"))])
#    from sklearn.dummy import DummyClassifier
#    clf = DummyClassifier(strategy="most_frequent")
#    base_learners_clas["majority"] = clf
    # base learners for regression problems
    base_learners_regr = OrderedDict()
    from sklearn.linear_model import Ridge, RidgeCV
#    base_learners_regr["ridge"] = Ridge(normalize=True)
    base_learners_regr["ridge_cv"] = RidgeCV(alphas=np.logspace(-1, 1, 5),
                                             normalize=True)
    
    # scoring measures for classification problems
    measures_clas = []
    measures_clas.append("CA")
    measures_clas.append("AUC")
    # scoring measures for regression problems
    measures_regr = []
    measures_regr.append("MAE")
    measures_regr.append("MSE")
    measures_regr.append("RMSE")
    measures_regr.append("Explained variance")
    
    learners = OrderedDict()
    learners["NoMerging"] = learning.NoMergingLearner()
    learners["MergeAll"] = learning.MergeAllLearner()
    no_filter = prefiltering.NoFilter()
    learners["ERM"] = learning.ERMLearner(folds=5, seed=33, prefilter=no_filter,
                                          error_func=squared_error)
    
    if test_config == 1:
        tasks_data = data.load_usps_digits_data()
        rnd_seed = 51
        repeats = 10
        results_path = os.path.join(path_prefix, "results/usps_digits-"
                                "seed{}-repeats{}".format(rnd_seed, repeats))
        test_tasks(tasks_data, results_path, base_learners_clas,
                   measures_clas, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.5, repeats=repeats)

    if test_config == 2:
        tasks_data = data.load_usps_digits_data()
        rnd_seed = 51
        repeats = 10
        subtasks_split = (3, 5)
        results_path = os.path.join(path_prefix, "results/usps_digits-"
                                "seed{0}-repeats{1}-subtasks{2[0]}_{2[1]}".\
                                format(rnd_seed, repeats, subtasks_split))
        test_tasks(tasks_data, results_path, base_learners_clas,
                   measures_clas, learners, "subtasks_split", rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.5, subtasks_split=subtasks_split,
                   repeats=repeats)
    
    if test_config == 3:
        tasks_data = data.load_usps_digits_data()
        rnd_seed = 51
        repeats = 10
        subtasks_split = (5, 10)
        results_path = os.path.join(path_prefix, "results/usps_digits-"
                                "seed{0}-repeats{1}-subtasks{2[0]}_{2[1]}".\
                                format(rnd_seed, repeats, subtasks_split))
        test_tasks(tasks_data, results_path, base_learners_clas,
                   measures_clas, learners, "subtasks_split", rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.5, subtasks_split=subtasks_split,
                   repeats=repeats)
    
    if test_config == 4:
        tasks_data = data.load_mnist_digits_data()
        rnd_seed = 51
        repeats = 10
        results_path = os.path.join(path_prefix, "results/mnist_digits-"
                                "seed{}-repeats{}".format(rnd_seed, repeats))
        test_tasks(tasks_data, results_path, base_learners_clas,
                   measures_clas, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.5, repeats=repeats)
    
    if test_config == 5:
        tasks_data = data.load_mnist_digits_data()
        rnd_seed = 51
        repeats = 10
        subtasks_split = (3, 5)
        results_path = os.path.join(path_prefix, "results/mnist_digits-"
                                "seed{0}-repeats{1}-subtasks{2[0]}_{2[1]}".\
                                format(rnd_seed, repeats, subtasks_split))
        test_tasks(tasks_data, results_path, base_learners_clas,
                   measures_clas, learners, "subtasks_split", rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.5, subtasks_split=subtasks_split,
                   repeats=repeats)
    
    if test_config == 6:
        tasks_data = data.load_mnist_digits_data()
        rnd_seed = 51
        repeats = 10
        subtasks_split = (5, 10)
        results_path = os.path.join(path_prefix, "results/mnist_digits-"
                                "seed{0}-repeats{1}-subtasks{2[0]}_{2[1]}".\
                                format(rnd_seed, repeats, subtasks_split))
        test_tasks(tasks_data, results_path, base_learners_clas,
                   measures_clas, learners, "subtasks_split", rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.5, subtasks_split=subtasks_split,
                   repeats=repeats)
    
    if test_config == 7:
        tasks_data = data.load_school_data()
        rnd_seed = 63
        repeats = 10
        results_path = os.path.join(path_prefix, "results/school-seed{}-"
                            "repeats{}".format(rnd_seed, repeats))
        test_tasks(tasks_data, results_path, base_learners_regr,
                   measures_regr, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.25, repeats=repeats,
                   weighting="task_sizes", error_margin="std")
    
    if test_config == 8:
        tasks_data = data.load_school_data()
        rnd_seed = 63
        repeats = 10
        results_path = os.path.join(path_prefix, "results/school-seed{}-"
                        "repeats{}-train_test_60-40".format(rnd_seed, repeats))
        test_tasks(tasks_data, results_path, base_learners_regr,
                   measures_regr, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.40, repeats=repeats,
                   weighting="task_sizes", error_margin="std")
    
    if test_config == 9:
        tasks_data = data.load_school_data()
        rnd_seed = 61
        repeats = 3
        keep = 10
        results_path = os.path.join(path_prefix, "results/school-seed{}-"
                            "repeats{}-keep{}".format(rnd_seed, repeats, keep))
        test_tasks(tasks_data, results_path, base_learners_regr,
                   measures_regr, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.25, repeats=repeats, keep=keep,
                   weighting="task_sizes")
    
    if test_config == 10:
        tasks_data = data.load_computer_survey_data()
        rnd_seed = 71
        repeats = 10
        results_path = os.path.join(path_prefix, "results/computer_survey-"
                            "seed{}-repeats{}".format(rnd_seed, repeats))
        test_tasks(tasks_data, results_path, base_learners_regr,
                   measures_regr, learners, "train_test_split",
                   rnd_seed=rnd_seed,
                   test=test, unpickle=unpickle, visualize=visualize,
                   test_prop=0.25, repeats=repeats,
                   weighting="task_sizes", error_margin="std")
