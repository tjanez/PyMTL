#
# stat.py
# Contains auxiliary methods for computing various statistical quantities.
#
# Copyright (C) 2010, 2011, 2012 Tadej Janez
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

import numpy
from scipy import stats

def mean(matrix):
    """Compute the means of columns of the given matrix.
    If the matrix is a column (i.e. only has one dimension), compute its mean.
    Return a numpy.array (for two-dimensional matrices) or a float (for
    one-dimensional matrices).
    
    Keyword arguments:
    matrix -- numpy.array (if matrix is not an array, a conversion will be
        attempted by NumPy)
    
    """
    return numpy.mean(matrix, axis=0)

def unbiased_std(matrix):
    """Compute the unbiased std. deviations of columns of the given matrix.
    If the matrix is a column (i.e. only has one dimension), compute its std.
    deviation.
    Return a numpy.array (for two-dimensional matrices) or a float (for
    one-dimensional matrices).
    
    Keyword arguments:
    matrix -- numpy.array (if matrix is not an array, a conversion will be
        attempted by NumPy)
    
    """
    # numpy.std() by default computes the biased std. dev. estimate;
    # "ddof=1" gives the unbiased estimator of std. dev. since the
    # divisor "N - ddof" is used
    return numpy.std(matrix, axis=0, ddof=1)

def _std_err(matrix):
    """Compute the std. errors of the means of columns of the given matrix.
    If the matrix is a column (i.e. only has one dimension), compute the std.
    error of its mean.
    Return a numpy.array (for two-dimensional matrices) or a float (for
    one-dimensional matrices).
    
    Keyword arguments:
    matrix -- numpy.array (if matrix is not an array, a conversion will be
        attempted by NumPy)
    
    """
    # using "ddof=1" gives the unbiased estimator of std. dev. 
    return stats.sem(matrix, axis=0, ddof=1)

def ci95(matrix):
    """Compute the 95% confidence intervals of the means of columns of the given
    matrix.
    If the matrix is a column (i.e. only has one dimension), compute the 95%
    confidence interval of its mean.
    Return a numpy.array where each entry corresponds to half of the length of
    the confidence interval of the mean of the column (i.e. 1.96 * std. error of
    the mean of the column).
    
    Keyword arguments:
    matrix -- numpy.array (if matrix is not an array, a conversion will be
        attempted by NumPy)
    
    """
    return 1.96 * _std_err(matrix)

if __name__ == "__main__":
    print mean([1, 2, 3, 4, 5, 6])
    print mean([[1, 2, 4], [4, 5, 6]])
    print unbiased_std([0.113967640255, 0.223095775796, 0.283134228235,
                        0.416793887842])
    print unbiased_std([[1, 2, 4], [4, 5, 6]])
    print _std_err([1, 2, 3, 4, 5, 6])
    print _std_err(([[1, 2, 4], [4, 5, 6]]))
    print ci95([1, 2, 3, 4, 5, 6])
    print ci95(([[1, 2, 4], [4, 5, 6]]))
    print ci95([])
    
