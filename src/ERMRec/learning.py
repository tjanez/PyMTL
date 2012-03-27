#
# learning.py
# Contains classes and methods implementing the merging learning methods.
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

import Orange

class MergeAllLearner:
    
    """Learning strategy that merges all users, regardless of whether they
    belong to the same behavior class or not. 
    
    """
    
    def __call__(self, users, base_learner):
        """Run the merging algorithm for the given users. Learn a single model
        on the merger of all users' data tables using the given base learner.
        Return a dictionary mapping from users' ids to the learned models (in
        this case, all users' ids will map to the same model).
        
        Keyword arguments:
        users -- a dictionary mapping from users' ids to their User objects
        base_learner -- an Orange learner
        
        """
        merged_data = None
        user_models = dict()
        for user_id, user in users.iteritems():
            if merged_data == None:
                merged_data = Orange.data.Table(user.get_learn_data())
            else:
                merged_data.extend(user.get_learn_data())
        model = base_learner(merged_data)
        for user_id in users.iterkeys():
            user_models[user_id] = model
        return user_models

class NoMergingLearner:
    
    """Learning strategy that doesn't merge any users. The learning algorithm
    only uses the data of each user to build its particular model.
    
    """
    
    def __call__(self, users, base_learner):
        """Run the merging algorithm for the given users. Learn a model using
        the given base learner for each user on its own data (no merging).
        Return a dictionary mapping from users' ids to the learned models.
        
        Keyword arguments:
        users -- a dictionary mapping from users' ids to their User objects
        base_learner -- an Orange learner
        
        """
        user_models = dict()
        for user_id, user in users.iteritems():
            user_models[user_id] = base_learner(user.get_learn_data())
        return user_models
    