#
# data.py
# Contains methods for loading users' data.
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

import numpy as np
from sklearn.datasets.base import Bunch

def _yes_no_converter(s):
    if s == "yes":
        return 1.
    elif s == "no":
        return 0.
    else:
        raise ValueError("Yes/No converter received an unexpected value: {}".\
                         format(s))

def _like_dislike_converter(s):
    if s == "like":
        return 1.
    elif s == "dislike":
        return 0.
    else:
        raise ValueError("Like/Dislike converter received an unexpected value: "
                         "{}".format(s))

def load_ratings_dataset(file_name):
    """Load an Orange tab file with user's data. Assumption is that the file
    has a particular number and order of features.

    This function uses numpy.genfromtxt to do most of the heavy-lifting.

    Parameters
    ----------
    file_name : {string}
        Path name of the data file to load.

    Returns
    -------
    data : Bunch
        A dict-like object holding
        "data": sample vectors
        "DESCR": the dataset's name/description
        "target": numeric classification labels (indices into the following)
        "target_names": symbolic names of classification labels

    """
    # columns 0 - 154 represent features, column 154 represents the class
    usecols = range(155)
    # specify converters for features and the class
    converters = {}
    # first two columns represent year and length of the movie
    # columns 2 - 130 represent actors
    for i in range(2, 130):
        converters[i] = _yes_no_converter
    # column 130 represents number of frequent actors
    # columns 131 - 154 represent genres
    for i in range(131, 154):
        converters[i] = _yes_no_converter
    # column 154 represents rating (class)
    converters[154] = _like_dislike_converter
    d = np.genfromtxt(file_name, dtype=float, delimiter='\t', skip_header=3,
                       converters=converters, missing_values='?',
                       usecols=usecols)
    X = np.array(d[:, :-1], dtype=float)
    y = np.array(d[:, -1], dtype=int)
    
    with open(file_name) as f:
        feature_names = f.readline().split('\t')[:155]
    
    return Bunch(data=X,
                 DESCR="Ratings for " + file_name[:-4],
                 target=y,
                 target_names=["dislike", "like"],
                 feature_names=feature_names)

if __name__ == "__main__":
    data = load_ratings_dataset("/home/tadej/Workspace/ERMRec/data/users-test3/"
                                "user00024.tab")
    print data.data[0]
