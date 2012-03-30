#
# orange_learners.py
# Contains custom wrappers for Orange machine learning algorithms.
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

import math

from Orange.classification.tree import C45Learner

class CustomC45Learner(C45Learner):
    
    """A version of the Orange's C45Learner with some additional custom
    parameters.
    
    """
    
    def __init__(self, min_objs_prop, **kwargs):
        """Set up a custom version of the C45Learner with an additional
        min_objs_prop parameter.
        All other keyword arguments are passed to the super class' __init__
        method.
        
        Arguments:
        min_objs_prop -- floating number between 0 and 1 representing the 
            minimal number of objects (instances) in leaves as a proportion
            of the size of the learning data (i.e. min_objs parameter will be
            set to min_objs_prop * len(learning data))
        
        """
        super(CustomC45Learner, self).__init__(**kwargs)
        if not (0 <= min_objs_prop <= 1):
            raise ValueError("Value of 'min_objs_prop' must be between 0 and 1")
        self.min_objs_prop = min_objs_prop
    
    def __call__(self, instances, *args, **kwargs):
        """Set the min_objs parameter of the C45Learner to min_objs_prop *
        len(instances) and pass all the arguments and keyword arguments to the
        super class' __call__ method.
        
        instances -- Orange data table
        
        """
        self.base.minObjs = int(math.ceil(self.min_objs_prop * len(instances)))
        return super(CustomC45Learner, self).__call__(instances, *args,
                                                      **kwargs)

if __name__ == "__main__":
    import time, Orange
    start = time.clock()
    data = Orange.data.Table("titanic")
#    data = Orange.data.Table("/home/tadej/Temp/merged_data_fold0.tab")
    end = time.clock()
    print "Data loading time: {:.3f}s".format(end-start)
    start = time.clock()
    l = CustomC45Learner(min_objs_prop=0.01, cf=25)
    c = l(data)
    end = time.clock()
    print "Classifier training time: {:.3f}s".format(end-start)
    