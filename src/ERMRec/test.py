#
# test.py
# Contains classes and methods for testing and comparing various machine
# learning algorithms on different sets of users and their ratings.
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

from ERMRec import stat
from ERMRec.data import load_ratings_dataset
from ERMRec.learning import prefiltering, learning
from ERMRec.plotting import BarPlotDesc, LinePlotDesc, plot_multiple

def configure_logging(name=None, level=logging.DEBUG,
                      console_level=logging.DEBUG,
                      file_name=None, file_level=logging.DEBUG):
    """Configure logging for the test module of ERMRec.
    Return the created Logger instance.
    
    Keyword arguments:
    name -- string representing the logger's name;
        if None, logging module will default to root logger
    level -- level of the created logger
    console_level -- level of the console handler attached to the created logger
    file_name -- file name of the file handler attached to the created logger;
        if None, no file handler is created 
    file_level -- level of the file handler attached to the created logger
    
    """
    # set up logging
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create formatter
    formatter = logging.Formatter(fmt="[%(asctime)s] %(name)-15s "
                    "%(levelname)-7s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # create console handler and configure it
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create a file handler and set its level
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

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

class User:
    
    """Contains data pertaining to a particular user and methods for extracting
    and manipulating this data.
    
    """
    def __init__(self, id, data):
        """Initialize a User object. Store the user's id and its data to private
        attributes.
        
        Arguments:
        id -- string representing user's id
        data -- sklearn.datasets.Bunch object holding user's data
        
        """
        self.id = id
        self._data = data
        self._active_fold = None
    
    def __str__(self):
        """Return a "pretty" representation of the user by indicating its id."""
        return self.id
    
    def get_data_size(self):
        """Return the number of examples in the user's data object"""
        return self._data.data.shape[0]
    
    def divide_data_into_folds(self, k, rand_seed):
        """Divide the user's data into the given number of folds.
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
        """Return a SHA1 hex digest based on user's id, data and
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
        third key corresponding to the user's id,
        fourth key corresponding to the scoring measure's name,
        value corresponding to the average value of the scoring measure.
    
    Keyword arguments:
    fold_scores -- five-dimensional dictionary with:
        first key corresponding to the fold number,
        second key corresponding to the base learner's name,
        third key corresponding to the learner's name,
        fourth key corresponding to the user's id,
        fifth key corresponding to the scoring measure's name,
        value corresponding to the scoring measure's value.
    
    """
    avg_scores = dict()
    for bl in fold_scores[0]:
        avg_scores[bl] = dict()
        for l in fold_scores[0][bl]:
            avg_scores[bl][l] = dict()
            for user_id in fold_scores[0][bl][l]:
                avg_scores[bl][l][user_id] = dict()
                for m_name in fold_scores[0][bl][l][user_id]:
                    u_scores = []
                    for i in fold_scores:
                        u_score = fold_scores[i][bl][l][user_id][m_name]
                        if u_score != None:
                            u_scores.append(u_score)
                    # the number of scores for each user is not always the
                    # same since it could happen that in some folds a
                    # scoring measures could not be computed
                    avg_scores[bl][l][user_id][m_name] = (sum(u_scores) /
                                                            len(u_scores))
    return avg_scores

class TestingResults:

    """Contains data of testing a particular base learning method on the pool
    of users.
    
    """
    
    def __init__(self, name, user_hashes, avg_scores):
        """Initialize a TestingResults object. Store the given arguments as
        attributes.
        
        Arguments:
        name -- string representing the base learner's name
        user_hashes -- OrderedDictionary with keys corresponding to users' ids
            and values to users' hashes
        avg_scores -- three-dimensional dictionary with:
            first key corresponding to the learner's name,
            second key corresponding to the user's id,
            third key corresponding to the scoring measure's name,
            value corresponding to the average value of the scoring measure
        
        """
        self.name = name
        self.user_hashes = user_hashes
        self.avg_scores = avg_scores
        

class UsersPool:
    
    """Contains methods for testing various learning algorithms on the given
    pool of users.
    
    """
    
    def __init__(self, users_data_path, seed):
        """Find all users who have data files in the given directory.
        Load data tables from these data files and create a new User object for
        each user.
        Create a dictionary mapping from users' ids to their User objects and
        store it in the self._users variable.
        Create a private Random object with the given seed and store it in the
        self._random variable.
        
        Keyword arguments:
        users_data_path -- string representing the path to the directory where
            users' ids and .tab files are stored
        seed -- integer to be used as a seed for the private Random object
        
        """
        self._users = OrderedDict()
        for file_ in sorted(os.listdir(users_data_path)):
            match = re.search(r"^user(\d+)\.tab$", file_)
            if match:
                # get the first parenthesized subgroup of the match
                user_id = match.group(1)
                data = load_ratings_dataset(os.path.join(users_data_path,
                                                         file_))
                user = User(user_id, data)
                self._users[user_id] = user
        self._random = random.Random(seed)
        # dictionary that will hold the TestingResults objects, one for each
        # tested base learner
        self._test_res = OrderedDict()
    
    def only_keep_k_users(self, k):
        """Reduce the size of the pool to k randomly chosen users.
        If the pool's size is smaller than k, keep all users.
        
        Arguments:
        k -- integer representing the number of users to keep
        
        """
        new_users = OrderedDict()
        for _ in range(min(k, len(self._users))):
            user_id = self._random.choice(self._users.keys())
            new_users[user_id] = self._users[user_id]
            del self._users[user_id]
        logger.info("Kept {} randomly chosen users, {} users discarded".\
                     format(len(new_users), len(self._users)))
        self._users = new_users
    
    def __str__(self):
        """Return a "pretty" representation of the pool of users by indicating
        their ids.
        
        """
        
        return "{} users: ".format(len(self._users)) + \
            ",".join(sorted(self._users.iterkeys()))
    
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
    
    def _find_bin_edge(self, n):
        """Find the appropriate bin edge for the given number of ratings.
        If the given value is smaller than the leftmost bin edge, an error is
        returned.
        If the given value is larger or equal to the rightmost bin edge, the
        rightmost bin edge is returned.
        
        Keyword arguments:
        n -- integer representing the number of ratings
        
        """
        i = bisect.bisect_right(self._bin_edges, n)
        if i <= 0:
            raise ValueError("The given number of ratings: '{}' is too small".\
                             format(n))
        return self._bin_edges[i-1]
    
    def divide_users_to_bins(self, bin_edges):
        """Look at the number of users' ratings and divide them into bins
        according to the given bin edges. No user should have less ratings than
        the leftmost bin edge.
        Store the given bin_edges to the self._bin_edges variable.
        Store the bins in a dictionary mapping from left bin edge to a list of
        users belonging to the corresponding bin (variable self._bins).
        
        Keyword arguments:
        bin_edges -- list of bin edges (should be sorted in ascending order)
         
        """
        self._bin_edges = bin_edges
        self._bins = {edge : [] for edge in self._bin_edges}
        for user_id, user in self._users.iteritems():
            # find the appropriate bin for the user
            bin_edge = self._find_bin_edge(user.get_data_size())
            self._bins[bin_edge].append(user_id)
        logger.debug("Divided the users into {} bins".format(len(
                                                            self._bin_edges)))
        logger.debug("Percent of users in each bin:")
        n_users = len(self._users)
        for i, bin_edge in enumerate(self._bin_edges[:-1]):
            logger.debug("{: >3}  --  {: >3}: {:.1f}%".format(bin_edge,
                self._bin_edges[i+1], 100.*len(self._bins[bin_edge])/n_users))
            if len(self._bins[bin_edge]) < 2:
                logger.warning("Bin '{: >3}--{: >3}' has less than 2 users".\
                    format(bin_edge, self._bin_edges[i+1]))
        logger.debug("{: >3}  --  {: >3}: {:.1f}%".format(self._bin_edges[-1],
                "inf", 100.*len(self._bins[self._bin_edges[-1]])/n_users))
    
    def _test_users(self, models, measures):
        """Test the given users' models on their testing data sets. Compute
        the given scoring measures of the testing results.
        Return a two-dimensional dictionary with the first key corresponding to
        the user's id and the second key corresponding to the measure's name.
        The value corresponds to the score for the given user and scoring
        measure.
        Note: If a particular scoring measure couldn't be computed for a user,
        its value is set to None.
        
        Keyword arguments:
        models -- dictionary mapping from users' ids to their models
        measures -- list of strings representing measure's names (currently,
            only CA and AUC are supported)
        
        """
        scores = dict()
        comp_errors = {measure : 0 for measure in measures}
        for user_id, user in self._users.iteritems():
            X_test, y_test = user.get_test_data()
            y_pred = models[user_id].predict(X_test)
            y_pred_proba = models[user_id].predict_proba(X_test)
            scores[user_id] = dict()
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
                scores[user_id][measure] = score
        # report the number of errors when computing the scoring measures
        n = len(self._users)
        for m_name, m_errors in comp_errors.iteritems():
            if m_errors > 0:
                logger.info("Scoring measure {} could not be computed for {}"
                    " out of {} users ({:.1f}%)".format(m_name, m_errors, n,
                    100.*m_errors/n))
        return scores
    
    def test_users(self, learners, base_learners, measures):
        """Divide all users' data into folds and perform the tests on each fold.
        Test the performance of the given learning algorithms with the given
        base learning algorithms and compute the testing results using the
        given scoring measures.
        Compute the average scores over all folds and store them in
        TestingResults objects, one for each base learner, along with users'
        hashes (used for comparing the results of multiple experiments).
        Store the created TestingResults objects in self._test_res, which is
        a dictionary with keys corresponding to base learner's names and values
        corresponding to their TestingResults objects. 
        
        Keyword arguments:
        learners -- ordered dictionary with items of the form (name, learner),
            where name is a string representing the learner's name and
            learner is a merging learning algorithm (e.g. ERM, NoMerging, ...) 
        base learners -- ordered dictionary with items of the form (name,
            learner), where name is a string representing the base learner's
            name and learner is an Orange learner
        measures -- list of strings representing measure's names (currently,
            only CA and AUC are supported)
        
        """
        # divide users' data into folds
        folds = 5
        for user in self._users.itervalues():
            user.divide_data_into_folds(folds, self._random.randint(0, 100))
        # perform learning and testing for each fold
        fold_scores = OrderedDict()
        for i in range(folds):
            for user in self._users.itervalues():
                user.set_active_fold(i)
            fold_scores[i] = {bl : dict() for bl in base_learners.iterkeys()}
            for bl in base_learners:
                for l in learners:
                    start = time.clock()
                    user_models = learners[l](self._users, base_learners[bl])
                    fold_scores[i][bl][l] = self._test_users(user_models,
                                                             measures)
                    end = time.clock()
                    logger.debug("Finished fold: {}, base learner: {}, "
                        "learner: {} in {:.2f}s".format(i, bl, l, end-start))
        # compute the average measure scores over all folds
        avg_scores = _compute_avg_scores(fold_scores)
        # get users' hashes
        user_hashes = OrderedDict()
        for user_id, user in self._users.iteritems():
            user_hashes[user_id] = user.get_hash()
        # store the average scores of each base learner in a separate
        # TestingResults object
        for bl in avg_scores:
            self._test_res[bl] = TestingResults(bl, user_hashes, avg_scores[bl])
    
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
        had the same users, users' data tables and cross-validation indices.
        In addition check if all TestingResults objects had the same learning
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
            # check if users' ids and hashes match for all base learning
            # algorithms
            if len(test_res_bl.user_hashes) != len(test_res_ref.user_hashes):
                return False
            for id, h in test_res_ref.user_hashes.iteritems():
                if test_res_bl.user_hashes[id] != h:
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
    
    def _compute_bin_stats(self, base_learner, learner, measure):
        """Compute the statistics (average, std. deviation and 95% confidence
        interval) of the performance of the given base learner and learner with
        the given measure for each bin of users in self._bins.
        Return a triple (avgs, stds, ci95s), where:
            avgs -- list of averages, one for each bin
            stds -- list of standard deviations, one for each bin
            ci95s -- list of 95% confidence intervals for the means, one for
                each bin
        
        """
        # prepare lists that will store the results
        avgs = []
        stds = []
        ci95s = []
        for bin_edge in self._bin_edges:
            # get the ids of users from the current bin
            bin = self._bins[bin_edge]
            # get the scores of users in the current bin for the given base
            # learner, learner and scoring measure
            bl_avg_scores = self._test_res[base_learner].avg_scores
            scores = np.array([bl_avg_scores[learner][id][measure] for id
                                  in bin])
            avgs.append(stat.mean(scores))
            stds.append(stat.unbiased_std(scores))
            ci95s.append(stat.ci95(scores))
        return avgs, stds, ci95s
        
    def visualize_results(self, base_learners, learners, measures, path_prefix,
                          colors, plot_type="line"):
        """Visualize the results of the given learning algorithms with the given
        base learning algorithms and the given scoring measures on the pool of
        users.
        Compute the averages, std. deviations and 95% conf. intervals on bins
        of users for all combinations of learners, base learners and scoring
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
        path_prefix -- string representing the prefix of the path where to save
            the generated plots
        colors -- dictionary mapping from learners' names to the colors that
            should represent them in the plots
            
        Keyword arguments:
        plot_type -- string indicating the type of the plot to draw (currently,
            only types "bar" and "line" are supported)
        
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
                    avgs, stds, ci95s = self._compute_bin_stats(bl, l, m)
                    if plot_type == "line":
                        plot_desc_sd[bl].append(LinePlotDesc(self._bin_edges,
                            avgs, stds, l, color=colors[l], ecolor=colors[l]))
                        plot_desc_ci95[bl].append(LinePlotDesc(self._bin_edges,
                            avgs, ci95s, l, color=colors[l], ecolor=colors[l]))
                    elif plot_type == "bar":
                        plot_desc_sd[bl].append(BarPlotDesc(self._bin_edges,
                            avgs, self._bin_edges[1] - self._bin_edges[0], stds,
                            l, color=colors[l], ecolor=colors[l]))
                        plot_desc_ci95[bl].append(BarPlotDesc(self._bin_edges,
                            avgs, self._bin_edges[1] - self._bin_edges[0],
                            ci95s, l, color=colors[l], ecolor=colors[l]))
                    else:
                        raise ValueError("Unsupported plot type: '{}'".\
                                         format(plot_type))
            plot_multiple(plot_desc_sd, results_path+"/{}-avg-SD.pdf".\
                    format(m), title="Avg. results for groups of users (error"
                    " bars show std. dev.)", subplot_title_fmt="Learner: {}",
                    xlabel="Number of ratings", ylabel=m)
            plot_multiple(plot_desc_ci95, results_path+"/{}-avg-CI.pdf".\
                    format(m), title="Avg. results for groups of users (error"
                    " bars show 95% conf. intervals)", subplot_title_fmt=\
                    "Learner: {}", xlabel="Number of ratings", ylabel=m)

if __name__ == "__main__":
    # a boolean indicating which pool of users to use
    test = True
    
    rnd_seed = 51
    # the number of users to keep in the users pool
    if test:
        keep = 10
    else:
        keep = 100
    
    # compute the location of other files/directories from the current file's
    # location
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    if test:
        users_data_path = os.path.join(path_prefix, "data/users-test2")
        results_path = os.path.join(path_prefix, "results/users-test2-seed{}-"
                                    "keep{}".format(rnd_seed, keep))
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    else:
        users_data_path = os.path.join(path_prefix, "data/users-m10")
        results_path = os.path.join(path_prefix, "results/users-m10-seed{}-"
                                    "keep{}".format(rnd_seed, keep))
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    pickle_path_fmt = os.path.join(results_path, "bl-{}.pkl")
    log_file = os.path.join(results_path, "run-{}.log".format(
                            time.strftime("%Y%m%d_%H%M%S")))
    
    # configure logging
    logger = configure_logging(name="ERMRec", console_level=logging.INFO,
                               file_name=log_file)
    
    # create a pool of users
    pool = UsersPool(users_data_path, rnd_seed)
    pool.only_keep_k_users(keep)
    # select base learners
    base_learners = OrderedDict()
    from sklearn.linear_model import LogisticRegression
#    from sklearn.pipeline import Pipeline
#    from sklearn_utils import MeanImputer
#    clf = Pipeline([("imputer", MeanImputer()),
#                    ("log_reg", LogisticRegression())])
    clf = LogisticRegression()
    base_learners["log_reg"] = clf
#    from sklearn.dummy import DummyClassifier
#    from sklearn.pipeline import Pipeline
#    from sklearn_utils import MeanImputer
#    clf = Pipeline([("imputer", MeanImputer()),
#                    ("majority", DummyClassifier(strategy="most_frequent"))])
#    base_learners["majority"] = clf
    #TODO: Replace or remove these Orange-based base learners
#    from orange_learners import CustomMajorityLearner
#    # a custom Majority learner which circumvents a bug with the  return_type
#    # keyword argument
#    base_learners["majority"] = CustomMajorityLearner()
#    base_learners["bayes"] = Orange.classification.bayes.NaiveLearner()
#    #base_learners["c45"] = Orange.classification.tree.C45Learner()
#    from orange_learners import CustomC45Learner
#    # custom C4.5 learner which allows us to specify the minimal number of
#    # examples in leaves as a proportion of the size of the data set
#    base_learners["c45_custom"] = CustomC45Learner(min_objs_prop=0.01)
#    # by default, Random Forest uses 100 trees in the forest and
#    # the square root of the number of features as the number of randomly drawn
#    # features among which it selects the best one to split the data sets in
#    # tree nodes
#    base_learners["rnd_forest"] = Orange.ensemble.forest.RandomForestLearner()
#    # by default, kNN sets parameter k to the square root of the numbers of
#    # instances
#    base_learners["knn"] = Orange.classification.knn.kNNLearner()
#    base_learners["knn5"] = Orange.classification.knn.kNNLearner(k=5)
#    from Orange.classification import svm
#    # these SVM parameters were manually obtained by experimenting in Orange
#    # Canvas using data in user02984.tab
#    base_learners["svm_RBF"] = svm.SVMLearner(svm_type=svm.SVMLearner.C_SVC,
#        kernel_type=svm.SVMLearner.RBF, C=100.0, gamma=0.01, cache_size=500)
    
    measures = []
    measures.append("CA")
    measures.append("AUC")
    
    learners = OrderedDict()
    learners["NoMerging"] = learning.NoMergingLearner()
    learners["MergeAll"] = learning.MergeAllLearner()
    no_filter = prefiltering.NoFilter()
    learners["ERM"] = learning.ERMLearner(folds=5, seed=33, prefilter=no_filter)
    
    # test all combinations of learners and base learners (compute the testing
    # results with the defined measures) and save the results
    pool.test_users(learners, base_learners, measures)
    pool.pickle_test_results(pickle_path_fmt)
    
    # find previously computed testing results and check if they were computed
    # using the same data tables and cross-validation indices
    pool.find_pickled_test_results(pickle_path_fmt)
    if not pool.check_test_results_compatible():
        raise ValueError("Test results for different base learners are not " \
                         "compatible.")
    # divide users into bins according to the number of ratings they have
    if test:
        bin_edges = [10, 15, 20]
    else:
        bin_edges = range(10, 251, 10)
    pool.divide_users_to_bins(bin_edges)
    
    # select the base learners, learners and scoring measures for which to 
    # visualize the testing results
    bls = pool.get_base_learners()
    ls = pool.get_learners()
    ms = pool.get_measures()
    pool.visualize_results(bls, ls, ms, results_path,
        colors={"NoMerging": "blue", "MergeAll": "green", "ERM": "red"},
        plot_type="line")
