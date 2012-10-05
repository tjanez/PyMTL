#
# test.py
# Contains unit tests for classes and methods in the test module.
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

import logging, os, shutil, unittest
from collections import OrderedDict

import Orange

from ERMRec import learning, orange_learners, test

# suppress logging of DEBUG and INFO messages to the console (not useful for
# running automated unit tests)
test.logger.handlers[0].setLevel(logging.WARNING)

# compute the location of other files/directories from the current file's
# location
cur_dir = os.path.dirname(os.path.abspath(__file__))
path_prefix = os.path.abspath(os.path.join(cur_dir, "../../../"))
users_data_path = os.path.join(path_prefix, "data/users-test")
results_path = os.path.join(path_prefix, "temp_unittest")
if not os.path.exists(results_path):
    os.makedirs(results_path)
pickle_path_fmt = os.path.join(results_path, "bl-{}.pkl")

class TestTestResultsCompatibilityChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a new UsersPool instance with a fixed random seed.
        Create new dictionaries with base learners, learners and measures.
        Test all combinations of learners and base learners and save the
        results to a file.
        
        """
        cls.rnd_seed = 51
        cls.pool = test.UsersPool(users_data_path, cls.rnd_seed)
        
        cls.base_learners = OrderedDict()
        cls.base_learners["majority"] = orange_learners.CustomMajorityLearner()
        
        cls.learners = OrderedDict()
        cls.learners["NoMerging"] = learning.learning.NoMergingLearner()
        cls.learners["MergeAll"] = learning.learning.MergeAllLearner()
        no_filter = learning.prefiltering.NoFilter()
        cls.learners["ERM"] = learning.learning.ERMLearner(folds=5, seed=33,
                                prefilter=no_filter)
        
        cls.measures = OrderedDict()
        cls.measures["CA"] = Orange.evaluation.scoring.CA
        cls.measures["AUC"] = Orange.evaluation.scoring.AUC
        
        cls.pool.test_users(cls.learners, cls.base_learners, cls.measures)
        cls.pool.pickle_test_results(pickle_path_fmt)
    
    @classmethod
    def tearDownClass(cls):
        """Remove the temporary results folder."""
        shutil.rmtree(results_path)

    def test_compatible_seed_learners_measures(self):
        """Create a new UsersPool instance with the same random seed as the
        one created in the setUpClass method.
        Create a new majority2 base learner to obtain some test results.
        Test the majority2 base learner with the compatible learners and
        measures.
        Check that the check_test_results_compatible function returns True.
        
        """
        new_pool = test.UsersPool(users_data_path, self.rnd_seed)
        
        new_base_learners = OrderedDict()
        new_base_learners["majority2"] = orange_learners.CustomMajorityLearner()
        
        new_pool.test_users(self.learners, new_base_learners, self.measures)
        new_pool.find_pickled_test_results(pickle_path_fmt)
        self.assertTrue(new_pool._test_res.has_key("majority"))
        self.assertTrue(new_pool.check_test_results_compatible())
    
    def test_incompatible_seeds(self):
        """Create a new UsersPool instance with an incompatible random seed.
        Create a new majority2 base learner to obtain some test results.
        Check that the check_test_results_compatible function returns False.
        
        """
        incompatible_rnd_seed = self.rnd_seed + 1
        new_pool = test.UsersPool(users_data_path, incompatible_rnd_seed)
        
        new_base_learners = OrderedDict()
        new_base_learners["majority2"] = orange_learners.CustomMajorityLearner()
        
        new_pool.test_users(self.learners, new_base_learners, self.measures)
        new_pool.find_pickled_test_results(pickle_path_fmt)
        self.assertTrue(new_pool._test_res.has_key("majority"))
        self.assertFalse(new_pool.check_test_results_compatible())
    
    def test_incompatible_number_of_learners(self):
        """Create a new UsersPool instance with the same random seed as the
        one created in the setUpClass method.
        Create a new majority2 base learner to obtain some test results.
        Test the majority2 base learner with just one learner and compatible
        measures.
        Check that the check_test_results_compatible function returns False.
        
        """
        new_pool = test.UsersPool(users_data_path, self.rnd_seed)
        
        new_base_learners = OrderedDict()
        new_base_learners["majority2"] = orange_learners.CustomMajorityLearner()
        
        new_learners = OrderedDict()
        new_learners["NoMerging"] = learning.learning.NoMergingLearner()
        
        new_pool.test_users(new_learners, new_base_learners, self.measures)
        new_pool.find_pickled_test_results(pickle_path_fmt)
        self.assertTrue(new_pool._test_res.has_key("majority"))
        self.assertFalse(new_pool.check_test_results_compatible())
    
    def test_incompatible_learners(self):
        """Create a new UsersPool instance with the same random seed as the
        one created in the setUpClass method.
        Create a new majority2 base learner to obtain some test results.
        Test the majority2 base learner with incompatible learners and
        compatible measures.
        Check that the check_test_results_compatible function returns False.
        
        """
        new_pool = test.UsersPool(users_data_path, self.rnd_seed)
        
        new_base_learners = OrderedDict()
        new_base_learners["majority2"] = orange_learners.CustomMajorityLearner()
        
        new_learners = OrderedDict()
        new_learners["NoMerging"] = learning.learning.NoMergingLearner()
        new_learners["MergeAll"] = learning.learning.MergeAllLearner()
        new_learners["FalseERM"] = learning.learning.NoMergingLearner()
        
        new_pool.test_users(new_learners, new_base_learners, self.measures)
        new_pool.find_pickled_test_results(pickle_path_fmt)
        self.assertTrue(new_pool._test_res.has_key("majority"))
        self.assertFalse(new_pool.check_test_results_compatible())
    
    def test_incompatible_number_of_measures(self):
        """Create a new UsersPool instance with the same random seed as the
        one created in the setUpClass method.
        Create a new majority2 base learner to obtain some test results.
        Test the majority2 base learner with compatible learners and just one
        measure.
        Check that the check_test_results_compatible function returns False.
        
        """
        new_pool = test.UsersPool(users_data_path, self.rnd_seed)
        
        new_base_learners = OrderedDict()
        new_base_learners["majority2"] = orange_learners.CustomMajorityLearner()
        
        new_measures = OrderedDict()
        new_measures["CA"] = Orange.evaluation.scoring.CA
        
        new_pool.test_users(self.learners, new_base_learners, new_measures)
        new_pool.find_pickled_test_results(pickle_path_fmt)
        self.assertTrue(new_pool._test_res.has_key("majority"))
        self.assertFalse(new_pool.check_test_results_compatible())
    
    def test_incompatible_measures(self):
        """Create a new UsersPool instance with the same random seed as the
        one created in the setUpClass method.
        Create a new majority2 base learner to obtain some test results.
        Test the majority2 base learner with compatible learners and
        incompatible measures.
        Check that the check_test_results_compatible function returns False.
        
        """
        new_pool = test.UsersPool(users_data_path, self.rnd_seed)
        
        new_base_learners = OrderedDict()
        new_base_learners["majority2"] = orange_learners.CustomMajorityLearner()
        
        new_measures = OrderedDict()
        new_measures["CA"] = Orange.evaluation.scoring.CA
        new_measures["FalseAUC"] = Orange.evaluation.scoring.CA
        
        new_pool.test_users(self.learners, new_base_learners, new_measures)
        new_pool.find_pickled_test_results(pickle_path_fmt)
        self.assertTrue(new_pool._test_res.has_key("majority"))
        self.assertFalse(new_pool.check_test_results_compatible())

if __name__ == '__main__':
    unittest.main()