#
# test.py
# Contains classes and methods for testing and comparing various machine
# learning algorithms on different sets of users and their ratings.
#
# Copyright (C) 2012 Tadej Janez
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

import bisect, os, random, re, time
import cPickle as pickle
from collections import OrderedDict

import numpy, Orange

from ERMRec.config import *
from ERMRec import stat
from ERMRec.learning import prefiltering, learning
from ERMRec.plotting import BarPlotDesc, plot_multiple

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
        """Initialize a User object. Store the user's id and its data table to
        private attributes.
        
        Arguments:
        id -- string representing user's id
        data -- Orange data table corresponding to the user
        
        """
        self.id = id
        self._data = data
        self._active_fold = None
    
    def __str__(self):
        """Return a "pretty" representation of the user by indicating its id."""
        return self.id
    
    def get_data_size(self):
        """Return the number of examples in the user's data table."""
        return len(self._data)
    
    def divide_data_into_folds(self, k, rand_seed):
        """Divide the user's data into the given number of folds.
        Store the random indices in the self._cv_indices variable.
        
        Keyword arguments:
        k -- integer representing the number of folds
        rand_seed -- integer representing the seed to use when initializing a
            local random number generator
        
        """
        self._k = k
        self._active_fold = None
        self._cv_indices = Orange.core.MakeRandomIndicesCV(self._data, k,
                stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                randseed=rand_seed)
    
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
        self._learn = self._data.select_ref(self._cv_indices, i, negate=True)
        self._test = self._data.select_ref(self._cv_indices, i)
    
    def get_learn_data(self):
        """Return the currently active learn data. """
        if self._active_fold == None:
            raise ValueError("There is no active fold!")
        return self._learn
    
    def get_test_data(self):
        """Return the currently active test data. """
        if self._active_fold == None:
            raise ValueError("There is no active fold!")
        return self._test
    
    def get_hash(self):
        """Return a hash value based on user's id, data table and
        cross-validation random indices. 
        
        """
        h1 = hash(self.id)
        # Note: Using "hash(self._data)" does not work as Orange gives different
        # hash values to data tables, even if they are loaded from the same
        # file;
        # A work-around is to use hash values of individual examples and merge
        # them together with left bit-shifts and xor
        h2 = 0
        for ex in self._data:
            h2 = ((h2 << 1) & 0xffffffff) ^ hash(ex)
        h3 = hash(tuple(self._cv_indices))
        return h1 ^ h2 ^ h3

def _compute_avg_scores(fold_scores):
    """Compute the average scores of the given fold scores.
    Return a four-dimensional dictionary with:
        first key corresponding to the base learner's name,
        second key corresponding to the learner's name,
        third key corresponding to the user's id,
        fourth key corresponding to the scoring measure's name,
        value corresponding to the average value of the scoring measure.
    
    Keyword arguments:
    fold_scores -- a five-dimensional dictionary with:
        first key corresponding to the fold number,
        second key corresponding to the base learner's name,
        third key corresponding to the learner's name,
        fourth key corresponding to the user's id,
        fifth key corresponding to the scoring measure's name,
        value corresponding to the scoring measure's value.
    
    """
    avg_scores = dict()
    for bl in fold_scores[0].iterkeys():
        avg_scores[bl] = dict()
        for l in fold_scores[0][bl].iterkeys():
            avg_scores[bl][l] = dict()
            for user_id in fold_scores[0][bl][l].iterkeys():
                avg_scores[bl][l][user_id] = dict()
                for m_name in fold_scores[0][bl][l][user_id].iterkeys():
                    u_scores = []
                    for i in fold_scores.iterkeys():
                        u_score = fold_scores[i][bl][l][user_id][m_name]
                        if u_score != None:
                            u_scores.append(u_score)
                    # the number of scores for each user is not always the
                    # same since it could happen that in some folds a
                    # scoring measures could not be computed
                    avg_scores[bl][l][user_id][m_name] = sum(u_scores) / \
                                                            len(u_scores)
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
        self._users = dict()
        for file in os.listdir(users_data_path):
            match = re.search(r"^user(\d+)\.tab$", file)
            if match:
                # get the first parenthesized subgroup of the match
                user_id = match.group(1)
                data_table = Orange.data.Table(os.path.join(users_data_path,
                                                            file))
                user = User(user_id, data_table)
                self._users[user_id] = user
        self._random = random.Random(seed)
        # dictionary that will hold the TestingResults objects, one for each
        # tested base learner
        self._test_res = OrderedDict()
    
    def __str__(self):
        """Return a "pretty" representation of the pool of users by indicating
        their ids."""
        
        return "{} users: ".format(len(self._users)) + \
            ",".join(sorted(self._users.iterkeys()))
    
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
        logging.debug("Divided the users into {} bins".format(len(
                                                            self._bin_edges)))
        logging.debug("Percent of users in each bin:")
        n_users = len(self._users)
        for i, bin_edge in enumerate(self._bin_edges[:-1]):
            logging.debug("{: >3}  --  {: >3}: {:.1f}%".format(bin_edge,
                self._bin_edges[i+1], 100.*len(self._bins[bin_edge])/n_users))
            if len(self._bins[bin_edge]) < 2:
                logging.warning("Bin '{: >3}--{: >3}' has less than 2 users".\
                    format(bin_edge, self._bin_edges[i+1]))
        logging.debug("{: >3}  --  {: >3}: {:.1f}%".format(self._bin_edges[-1],
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
        measures -- ordered dictionary with items of the form (name,
            measure), where name is a string representing the measure's name and
            measure is an Orange scoring measure (e.g. "CA", AUC", ...)
        
        """
        scores = dict()
        comp_errors = {m_name : 0 for m_name in measures.iterkeys()}
        for user_id, user in self._users.iteritems():
            results = Orange.evaluation.testing.test_on_data([models[user_id]],
                                user.get_test_data())
            scores[user_id] = dict()
            for m_name, m_func in measures.iteritems():
                m_scores = m_func(results)
                if m_scores == False:
                    # m_func returned False; probably AUC cannot be computed
                    # because all instances belong to the same class
                    m_score = None
                    comp_errors[m_name] += 1
                else:
                    m_score = m_scores[0]
                scores[user_id][m_name] = m_score
        # report the number of errors when computing the scoring measures
        n = len(self._users)
        for m_name, m_errors in comp_errors.iteritems():
            if m_errors > 0:
                logging.info("Scoring measure {} could not be computed for {}" \
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
        measures -- ordered dictionary with items of the form (name, measure),
            where name is a string representing the measure's name and measure
            is an Orange scoring measure (e.g. CA, AUC, ...)
        
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
                    logging.debug("Finished fold: {}, base learner: {}, " \
                        "learner: {} in {:.2f}s".format(i, bl, l, end-start))
        # compute the average measure scores over all folds
        avg_scores = _compute_avg_scores(fold_scores)
        # get users' hashes
        user_hashes = OrderedDict()
        for user_id, user in self._users.iteritems():
            user_hashes[user_id] = user.get_hash()
        # store the average scores of each base learner in a separate
        # TestingResults object
        for bl in avg_scores.iterkeys():
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
            logging.debug("Successfully pickled the results of base learner: " \
                          "{} to file: {}".format(bl, pickle_path))
            
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
            raise ValueError("The given pickle_path_fmt does not have an " \
                "appropriate format.")
        re_template = r"^{}(.+){}$".format(match.group(1), match.group(2)) 
        for file in sorted(os.listdir(dir_name)):
            match = re.search(re_template, file)
            if match:
                bl = match.group(1)
                if bl not in self._test_res:
                    file_path = os.path.join(dir_name, file)
                    tr = unpickle_obj(file_path)
                    if not isinstance(tr, TestingResults):
                        raise TypeError("Object loaded from file: {} is not " \
                            "of type TestingResults.".format(file_path))
                    self._test_res[bl] = tr
                    logging.debug("Successfully unpickled the results of base" \
                        " learner: {} from file: {}".format(bl, file_path))
                else:
                    logging.info("Results of base learner: {} are already " \
                        "loaded".format(bl))
        
    def check_test_results_compatible(self):
        """Check if all TestingResults objects in the self._test_res dictionary
        had the same users, users' data tables and cross-validation indices.
        
        """
        bls = tuple(self._test_res.iterkeys())
        if len(bls) <= 1:
            return True
        # select the first base learner as the reference base learner
        ref = bls[0]
        for user_id, hash in self._test_res[ref].user_hashes.iteritems():
            for bl in bls[1:]:
                if self._test_res[bl].user_hashes[user_id] != hash:
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
            scores = numpy.array([bl_avg_scores[learner][id][measure] for id
                                  in bin])
            avgs.append(stat.mean(scores))
            stds.append(stat.unbiased_std(scores))
            ci95s.append(stat.ci95(scores))
        return avgs, stds, ci95s
        
    def visualize_results(self, base_learners, learners, measures, path_prefix,
                          colors):
        """Visualize the results of the given learning algorithms with the given
        base learning algorithms and the given scoring measures.
        with the given scoring measures on the pool of users.
        Compute the averages, std. deviations and 95% conf. intervals on bins
        of users for all combinations of learners, base learners and scoring
        measures.
        Draw a big plot displaying the averages and std. deviations for each
        scoring measure. Each big plot has one subplot for each base learner.
        Each subplot shows the comparison between different learning algorithms.
        The same big plots are drawn for averages and 95% conf. intervals.
        Save the drawn plots to the files with the given path prefix.
        
        Keyword arguments:
        base_learners -- list of strings representing the names of base
            learners
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
                    avgs, stds, ci95s = self._compute_bin_stats(bl, l, m)
                    plot_desc_sd[bl].append(BarPlotDesc(self._bin_edges, avgs,
                        self._bin_edges[1] - self._bin_edges[0], stds, l,
                        color=colors[l], ecolor=colors[l]))
                    plot_desc_ci95[bl].append(BarPlotDesc(self._bin_edges, avgs,
                        self._bin_edges[1] - self._bin_edges[0], ci95s, l,
                        color=colors[l], ecolor=colors[l]))
            plot_multiple(plot_desc_sd, results_path+"/{}-avg-SD.pdf".\
                    format(m), title="Avg. results for groups of users (error" \
                    " bars show std. dev.)", subplot_title_fmt="Learner: {}",
                    xlabel="Number of ratings", ylabel=m)
            plot_multiple(plot_desc_ci95, results_path+"/{}-avg-CI.pdf".\
                    format(m), title="Avg. results for groups of users (error" \
                    " bars show 95% conf. intervals)", subplot_title_fmt=\
                    "Learner: {}", xlabel="Number of ratings", ylabel=m)

if __name__ == "__main__":
    # a boolean indicating which pool of users to use
    test = True
    
    # compute the location of other files/directories from the current file's
    # location
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    if test:
        users_data_path = os.path.join(path_prefix, "data/users-test")
        results_path = os.path.join(path_prefix, "results/users-test")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        pickle_path_fmt = os.path.join(results_path, "bl-{}.pkl")
    else:
        users_data_path = os.path.join(path_prefix, "data/users-m10")
        results_path = os.path.join(path_prefix, "results/users-m10new")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        pickle_path_fmt = os.path.join(results_path, "bl-{}.pkl")
        
    # create a pool of users
    rnd_seed = 51
    pool = UsersPool(users_data_path, rnd_seed)
    
    base_learners = OrderedDict()
#    base_learners["majority"] = Orange.classification.majority.MajorityLearner()
    base_learners["bayes"] = Orange.classification.bayes.NaiveLearner()
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
    
    measures = OrderedDict()
    measures["CA"] = Orange.evaluation.scoring.CA
    measures["AUC"] = Orange.evaluation.scoring.AUC
    
    learners = OrderedDict()
#    learners["NoMerging"] = learning.NoMergingLearner()
#    learners["MergeAll"] = learning.MergeAllLearner()
    no_filter = prefiltering.NoFilter()
    learners["ERM"] = learning.ERMLearner(folds=5, seed=33, prefilter=no_filter)
    
    # test all combinations of learners and base learners (compute the testing
    # results with the defined measures) and save the results
    pool.test_users(learners, base_learners, measures)
    pool.pickle_test_results(pickle_path_fmt)
    
#    # find previously computed testing results and check if they were computed
#    # using the same data tables and cross-validation indices
#    pool.find_pickled_test_results(pickle_path_fmt)
#    if not pool.check_test_results_compatible():
#        raise ValueError("Test results for different base learners are not " \
#                         "compatible.")
    # divide users into bins according to the number of ratings they have
    if test:
        bin_edges = [10, 15, 20]
    else:
        bin_edges = range(10, 251, 10)
    pool.divide_users_to_bins(bin_edges)
    
    pool.visualize_results(list(base_learners.iterkeys()),
        list(learners.iterkeys()), list(measures.iterkeys()), results_path,
        colors={"NoMerging": "blue", "MergeAll": "green", "ERM": "red"})
