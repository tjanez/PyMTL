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
        variance).
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
    # y contains elements from 1 to 10; convert them to proper digits from
    # 0 to 9
    y -= 1
    # convert this multi-way classification problem to a MTL problem
    # (as described in Kang et al., ICML 2011)
    tasks = []
    for c in np.unique(y):
        descr = "USPS digits data: {} vs. other classes".format(c)
        id = "{} vs. others".format(c)
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
        variance).
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
    # y contains elements from 1 to 10; convert them to proper digits from
    # 0 to 9
    y -= 1
    # convert this multi-way classification problem to a MTL problem
    # (as described in Kang et al., ICML 2011)
    tasks = []
    for c in np.unique(y):
        descr = "MNIST digits data: {} vs. other classes".format(c)
        id = "{} vs. others".format(c)
        cur_y = np.array(y, copy=True)
        cur_y[y == c] = 1
        cur_y[y != c] = 0
        tasks.append(Bunch(data=X,
                           target=cur_y,
                           target_names=["other", str(c)],
                           DESCR=descr,
                           ID=id))
    return tasks


def load_school_data():
    """Load School data set.
    
    The data set's properties are:
    - 15362 samples
    - 28 features (original features were:
        - year of examination,
        - 4 school-specific features,
        - 3 student-specific features.
        Categorical features were replaced with a binary feature for each
        possible feature value. In total this resulted in 27 features. The last
        feature is the bias term) 
    - regression task of predicting student's exam score
    
    Note
    ----
    The data set is the one provided by "Argyriou, Evgeniou, Pontil - Convex
    multi-task feature learning - ML 2008" on their web site.
    
    Returns
    -------
    tasks -- list
        A list of Bunch objects that correspond to regression tasks, each task
        corresponding to one school.
    
    """
    matlab_file = os.path.join(path_prefix, "data/school/school_b.mat")
    mat = loadmat(matlab_file)
    # extract combined X and y data
    X = mat["x"].T.astype("float")
    y = mat["y"].astype("float")
#    # DEBUG
#    print "Inspection of values of features for School data"
#    for i in range(X.shape[1]):
#        col = X[:, i]
#        if not np.all(np.unique(col) == np.array([0, 1])):
#            print "Column {} has non-binary unique values: {}".format(i, np.unique(col))
    # extract starting indices of tasks and subtract 1 since MATLAB uses 1-based
    # indexing
    start_ind = np.ravel(mat["task_indexes"] - 1)
    # split the data to separate tasks
    for i in range(len(start_ind)):
        start = start_ind[i]
        if i == len(start_ind) - 1:
            end = -1
        else:
            end = start_ind[i + 1]
        descr = "School data: school {}".format(i + 1)
        id = "School {}".format(i + 1)
        tasks.append(Bunch(data=X[start:end],
                           target=y[start:end],
                           DESCR=descr,
                           ID=id))

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
    print
    
    tasks = load_school_data()
