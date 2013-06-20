#
# bin_exp.py
# Contains classes and methods implementing special multi-task learning methods
# used in the binarization experiment.
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

import logging
from collections import OrderedDict

import numpy as np
import Orange

from PyMTL.util import logger


class TreeMarkedAndMergedLearner(object):
    
    """Special multi-task learning strategy that only works with the
    Orange's TreeLearner as the base learner and the data given in the Orange
    format.
    It is used in the binarization experiment.
    
    """
    
    def __call__(self, task_ids, merged_data, base_learner):
        """Check that the given base learner is an Orange TreeLearner and use
        it on the merged data to build a common model for all tasks.
        Assign the fitted model to all tasks.
        Return a dictionary of data structures computed within this learner.
        It has the following keys:
            task_models -- dictionary mapping from tasks' ids to the learned
                models (in this case, all tasks' ids will map to the same model)
        
        Arguments:
        task_ids -- list of tasks' ids
        merged_data -- Orange.data.Table representing the merged learning data
            of all tasks
        base_learner -- Orange.classification.tree.TreeLearner representing the
            base learner to build the models
        
        """
        # check that the given base learner is an Orange's TreeLearner
        if not isinstance(base_learner, Orange.classification.tree.TreeLearner):
            raise ValueError("The base_learner should be an Orange "
                             "TreeLearner.")
        # build a model on the merged data
        model = base_learner(merged_data)
        # assign the fitted model to all tasks
        task_models = dict()
        for tid in task_ids:
            task_models[tid] = model
        # create and fill the return dictionary
        R = dict()
        R["task_models"] = task_models
        return R


from PyMTL.orange_utils import ForcedFirstSplitTreeLearner

class ForcedFirstSplitMTLLearner(TreeMarkedAndMergedLearner):
    
    """A sub-class of the TreeMarkedAndMergedLearner which transforms the given
    TreeLearner base learner to a ForcedFirstSplitTreeLearner and uses it to
    build the model.
    It is used in the binarization experiment.
    
    Parameters
    ----------
    first_split_attr : str
        The name of the attribute to be used as the first split when building a
        decision tree.
    
    """
    def __init__(self, first_split_attr):
        self.first_split_attr = first_split_attr
    
    def __call__(self, task_ids, merged_data, base_learner):
        """Check that the given base learner is an Orange TreeLearner and then
        transform it into a ForcedFirstSplitTreeLearner. Use it on the merged
        data to build a common model for all tasks.
        Assign the fitted model to all tasks.
        Return a dictionary of data structures computed within this learner.
        It has the following keys:
            task_models -- dictionary mapping from tasks' ids to the learned
                models (in this case, all tasks' ids will map to the same model)
        
        Arguments:
        task_ids -- list of tasks' ids
        merged_data -- Orange.data.Table representing the merged learning data
            of all tasks
        base_learner -- Orange.classification.tree.TreeLearner representing the
            base learner to build the models
        
        """
        # check that the given base learner is an Orange's TreeLearner
        if not isinstance(base_learner, Orange.classification.tree.TreeLearner):
            raise ValueError("The base_learner should be an Orange "
                             "TreeLearner.")
        # create an instance of the ForcedFirstSplitTreeLearner with the same
        # attributes as the given base_learner
        ffstl = ForcedFirstSplitTreeLearner(first_split_attr=
                                            self.first_split_attr)
        for k, v in base_learner.__dict__.items():
            try:
                ffstl.__dict__[k] = v
            except:
                logger.debug("Could not set the value of attribute: {}".\
                             format(k))
        # build a model on the merged data
        model = ffstl(merged_data)
        # assign the fitted model to all tasks
        task_models = dict()
        for tid in task_ids:
            task_models[tid] = model
        # create and fill the return dictionary
        R = dict()
        R["task_models"] = task_models
        return R
