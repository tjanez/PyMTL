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

import bisect, os, random, re
import cPickle as pickle
from collections import OrderedDict

import numpy, Orange
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from ERMRec.config import *
from ERMRec import stat

def unpickle(file_path):
    """Unpickle a UsersPool object from a file.
    Return the reference to the unpickled object.
    
    Keyword arguments:
    file_path -- string representing the path to the file where the object is
        pickled
    
    """
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)

class UsersPool:
    
    """Contains methods for testing various learning algorithms on the given
    pool of users.
    
    """
    
    def __init__(self, users_data_path, seed):
        """Find all users who have data files in the given directory.
        Load data tables from these data files, create a dictionary mapping
        from users' ids to their data tables and store it in the
        self._data_tables variable.
        Create a private Random object with the given seed.
        
        Keyword arguments:
        users_data_path -- string representing the path to the directory where
            users' ids and .tab files are stored
        seed -- integer to be used as a seed for the private Random object
        
        """
        self._data_tables = dict()
        for file in os.listdir(users_data_path):
            match = re.search(r"^user(\d+)\.tab$", file)
            if match:
                # get the first parenthesized subgroup of the match
                user_id = match.group(1)
                data_table = Orange.data.Table(os.path.join(users_data_path,
                                                            file))
                self._data_tables[user_id] = data_table
        self._random = random.Random(seed)
    
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
        bin_edges -- a list of bin edges (should be sorted in ascending order)
         
        """
        self._bin_edges = bin_edges
        self._bins = {edge : [] for edge in self._bin_edges}
        for user_id, data in self._data_tables.iteritems():
            # find the appropriate bin for the user
            bin_edge = self._find_bin_edge(len(data))
            self._bins[bin_edge].append(user_id)
        logging.debug("Divided the users into {} bins".format(len(
                                                            self._bin_edges)))
        logging.debug("Percent of users in each bin:")
        n_users = len(self._data_tables)
        for i, bin_edge in enumerate(self._bin_edges[:-1]):
            logging.debug("{: >3}  --  {: >3}: {:.1f}%".format(bin_edge,
                self._bin_edges[i+1], 100.*len(self._bins[bin_edge])/n_users))
            if len(self._bins[bin_edge]) < 2:
                logging.warning("Bin '{: >3}--{: >3}' has less than 2 users".\
                    format(bin_edge, self._bin_edges[i+1]))
        logging.debug("{: >3}  --  {: >3}: {:.1f}%".format(self._bin_edges[-1],
                "inf", 100.*len(self._bins[self._bin_edges[-1]])/n_users))
    
    def _test_user(self, data, learners, measures):
        """Perform a cross-validation on the given data and compare the given
        learning algorithms with the given scoring measures.
        Return a two-dimensional dictionary with the first key corresponding to
        the learning algorithm's name and the second key corresponding to the 
        measure's name. The value corresponds to the score with the given
        learner and scoring measure.
        
        Keyword arguments:
        data -- an Orange data table with instances belonging to a user
        learners -- an ordered dictionary with items of the form (name,
            learner), where name is a string representing the learner's name and
            learner is an Orange learner
        measures -- an ordered dictionary with items of the form (name,
            measure), where name is a string representing the measure's name and
            measure is an Orange scoring measure (e.g. "CA", AUC", ...)
        
        """
        orange_learners = [l for l in learners.itervalues()]
        results = Orange.evaluation.testing.cross_validation(orange_learners,
                    data, folds=5, randseed=self._random.randint(0, 100))
        scores = {l_name : dict() for l_name in learners.iterkeys()}
        for m_name, m_func in measures.iteritems():
            m_scores = m_func(results)
            for (l_name, _), m_score in zip(learners.iteritems(), m_scores):
                scores[l_name][m_name] = m_score
        return scores
    
    def test_users(self, learners, measures):
        """Test the performance of the given learning algorithms with the given
        scoring measures on the pool of users.
        
        Keyword arguments:
        learners -- an ordered dictionary with items of the form (name,
            learner), where name is a string representing the learner's name and
            learner is an Orange learner
        measures -- an ordered dictionary with items of the form (name,
            measure), where name is a string representing the measure's name and
            measure is an Orange scoring measure (e.g. "CA", AUC", ...)
        
        """
        self._user_scores = dict()
        for user_id, data in self._data_tables.iteritems():
            res = self._test_user(data, learners, measures)
            logging.debug("Finished testing on data table of user {}".\
                          format(user_id))
            self._user_scores[user_id] = res
        return self._user_scores
    
    def _compute_scores(self, learner, measure):
        """Compute the average, std. deviation and 95% confidence interval for
        the performance of the given learner with the given measure. Compute
        these values for each bin of users in self._bins.
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
            # get the scores of users in the current bin for the given learner
            # and scoring measure
            scores = numpy.array([self._user_scores[id][learner][measure] for
                                  id in bin])
            avgs.append(stat.mean(scores))
            stds.append(stat.unbiased_std(scores))
            ci95s.append(stat.ci95(scores))
        return avgs, stds, ci95s
    
    def _bar_plot(self, left_edges, heights, yerr, file_name, color="blue",
                  ecolor="red", title="", xlabel="", ylabel="", label=""):
        """Draw a bar plot with the given left edges, heights and y error bars.
        Save the plot to the given file name.
        
        Keyword arguments:
        left_edges -- list of bars' left edges
        heights -- list of bars' heights
        yerr -- list of values representing the heights of the +/- error bars
        file_name -- string representing the name of the file, where to save
            the drawn plot
        color -- string representing the color of the bars
        ecolor -- string representing the color of error bars
        title -- string representing plot's title
        xlabel -- string representing x axis's label
        xlabel -- string representing y axis's label
        label -- string representing bar plot's label to be used in the legend
        
        """
        width = self._bin_edges[1] - self._bin_edges[0]
        plt.bar(left_edges, heights, width=width, yerr=yerr, color=color,
                ecolor=ecolor, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(0.0, 1.0)
        plt.xlim(xmin=0.0)
        plt.grid(b=True)
        plt.legend(loc="upper right", fancybox=True,
                   prop=FontProperties(size="small"))
        plt.savefig(file_name)
        plt.clf()
        
    def visualize_results(self, learners, measures, path_prefix):
        """Visualize the results of the given machine learning algorithms
        with the given scoring measures on the pool of users.
        For each combination of machine learning algorithm and scoring measure,
        draw a bar plot showing means and std. deviations of each bin of users.
        And the same for means and 95% conf. intervals for the means.
        Save the drawn plots to the files with the given path prefix.
        
        Keyword arguments:
        learners -- a list of strings representing names of the machine learning
            algorithms
        measures -- a list of strings representing names of the scoring measures
        path_prefix -- string representing the prefix of the path where to save
            the generated plots
        
        """
        for m in measures:
            for l in learners:
                avgs, stds, ci95s = self._compute_scores(l, m)
                self._bar_plot(self._bin_edges, avgs, stds, path_prefix + \
                    "-{}-{}-avg-SD.pdf".format(m, l), title="Avg. results for" \
                    " groups of users (error bars show std. dev.)", xlabel=\
                    "Number of ratings", ylabel=m, label=l)
                self._bar_plot(self._bin_edges, avgs, ci95s, path_prefix + \
                    "-{}-{}-avg-95CI.pdf".format(m, l), title="Avg. results " \
                    "for groups of users (error bars show 95% CI)", xlabel=\
                    "Number of ratings", ylabel=m, label=l)
    
    def pickle(self, file_path):
        """Pickle yourself to the given file_path.
        
        Keyword arguments:
        file_path -- string representing the path to the file where to pickle
            the object
        
        """
        with open(file_path, "wb") as pkl_file:
            pickle.dump(self, pkl_file, pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    # a boolean indicating which pool of users to use
    test = True
    
    # compute the location of other files/directories from the current file's
    # location
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    if test:
        users_data_path = os.path.join(path_prefix, "data/users-test")
        pickle_path = os.path.join(path_prefix, "results/users-test.pkl")
        results_prefix = os.path.join(path_prefix, "results/users-test")
    else:
        users_data_path = os.path.join(path_prefix, "data/users-m10")
        pickle_path = os.path.join(path_prefix, "results/users-m10.pkl")
        results_prefix = os.path.join(path_prefix, "results/users-m10")
        
    # create a pool of users
    rnd_seed = 51
    pool = UsersPool(users_data_path, rnd_seed)
    # divide users into bins according to the number of ratings they have
    if test:
        bin_edges = [10, 15, 20]
    else:
        bin_edges = range(10, 251, 10)
    pool.divide_users_to_bins(bin_edges)
    
    learners = OrderedDict()
#    learners["bayes"] = Orange.classification.bayes.NaiveLearner()
#    learners["c45"] = Orange.classification.tree.C45Learner()
    # by default, Random Forest uses 100 trees in the forest and
    # the square root of the number of features as the number of randomly drawn
    # features among which it selects the best one to split the data sets in
    # tree nodes
#    learners["rnd_forest"] = Orange.ensemble.forest.RandomForestLearner()
    # by default, kNN sets parameter k to the square root of the numbers of
    # instances
#    learners["knn"] = Orange.classification.knn.kNNLearner()
#    learners["knn5"] = Orange.classification.knn.kNNLearner(k=5)
    from Orange.classification import svm
    # these SVM parameters were manually obtained by experimenting in Orange
    # Canvas using data in user02984.tab
    learners["svm_RBF"] = svm.SVMLearner(svm_type=svm.SVMLearner.C_SVC,
        kernel_type=svm.SVMLearner.RBF, C=100.0, gamma=0.01, cache_size=500)
    
    measures = OrderedDict()
    measures["CA"] = Orange.evaluation.scoring.CA
    measures["AUC"] = Orange.evaluation.scoring.AUC
    
    pool.test_users(learners, measures)
    pool.visualize_results(list(learners.iterkeys()), list(measures.iterkeys()),
        results_prefix)
    pool.pickle(pickle_path)
#    pool = unpickle(pickle_path)
#    pool.test_users(learners, measures)
