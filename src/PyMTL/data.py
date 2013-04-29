#
# data.py
# Contains methods for loading various multi-task learning (MTL) data sets.
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

import os.path

import numpy as np
from scipy.io import loadmat
from sklearn.datasets.base import Bunch

# find out the current file's location so it can be used to compute the
# location of other files/directories
cur_dir = os.path.dirname(os.path.abspath(__file__))
path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))

def load_usps_digits_data():
    """Load the USPS digits data set. Convert it from the original multi-way
    classification problem to a MTL problem. 
    
    The data set's properties are:
    - 2000 samples
    - 64 features (original images were pre-processed with PCA and their
        dimensionality was reduced to 64, which retains ~95% of the total
        variance.
    - 10 classes, one for each of the 10 digits
    
    Note
    ----
    The data set is the one provided by "Kang, Grauman, Sha - Learning with Whom
    to Share in Multi-task Feature Learning - ICML 2011" on their web site.
    
    Returns
    -------
    tasks -- list
        A list of Bunch objects that correspond to binary classification tasks
        of one digit against all the others.
    
    """
    matlab_file = os.path.join(path_prefix, "data/usps_digits/"
                               "split1_usps_1000train.mat")
    mat = loadmat(matlab_file)
    # combine training, validation and testing arrays into one big array
    X = np.concatenate((mat["digit_trainx"], mat["digit_validx"],
                        mat["digit_testx"]), axis=0)
    y = np.concatenate((mat["digit_trainy"], mat["digit_validy"],
                        mat["digit_testy"]), axis=0)
    # convert this multi-way classification problem to a MTL problem
    # (as described in Kang et al., ICML 2011)
    tasks = []
    for c in np.unique(y):
        descr = "USPS digits data: {} vs. other classes".format(c)
        id = "{} vs. other".format(c)
        cur_y = np.array(y, copy=True)
        cur_y[y == c] = 1
        cur_y[y != c] = 0
        tasks.append(Bunch(data=X,
                           target=cur_y,
                           target_names=["other", str(c)],
                           DESCR=descr,
                           ID=id))
    return tasks


def load_mnist_digits_data():
    """Load the MNIST digits data set. Convert it from the original multi-way
    classification problem to a MTL problem. 
    
    The data set's properties are:
    - 2000 samples
    - 87 features (original images were pre-processed with PCA and their
        dimensionality was reduced to 87, which retains ~95% of the total
        variance.
    - 10 classes, one for each of the 10 digits
    
    Note
    ----
    The data set is the one provided by "Kang, Grauman, Sha - Learning with Whom
    to Share in Multi-task Feature Learning - ICML 2011" on their web site.
    
    Returns
    -------
    tasks -- list
        A list of Bunch objects that correspond to binary classification tasks
        of one digit against all the others.
    
    """
    matlab_file = os.path.join(path_prefix, "data/mnist_digits/"
                               "split1_mnist_1000train.mat")
    mat = loadmat(matlab_file)
    # combine training, validation and testing arrays into one big array
    X = np.concatenate((mat["digit_trainx"], mat["digit_validx"],
                        mat["digit_testx"]), axis=0)
    y = np.concatenate((mat["digit_trainy"], mat["digit_validy"],
                        mat["digit_testy"]), axis=0)
    # convert this multi-way classification problem to a MTL problem
    # (as described in Kang et al., ICML 2011)
    tasks = []
    for c in np.unique(y):
        descr = "MNIST digits data: {} vs. other classes".format(c)
        id = "{} vs. other".format(c)
        cur_y = np.array(y, copy=True)
        cur_y[y == c] = 1
        cur_y[y != c] = 0
        tasks.append(Bunch(data=X,
                           target=cur_y,
                           target_names=["other", str(c)],
                           DESCR=descr,
                           ID=id))
    return tasks


if __name__ == "__main__":
    tasks = load_usps_digits_data()
    print "Loaded the USPS digits MTL problem with the {} tasks:".\
        format(len(tasks))
    for t in tasks:
        print t.DESCR
    print
    
    tasks = load_mnist_digits_data()
    print "Loaded the MNIST digits MTL problem with the {} tasks:".\
        format(len(tasks))
    for t in tasks:
        print t.DESCR
