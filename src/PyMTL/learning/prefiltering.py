#
# prefiltering.py
# Contains classes and methods for pre-filtering pairs of users in the ERM
# merging learning method.
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

class NoFilter:
    
    """Pre-filtering method that doesn't filter out any pair of users. """
    
    def __call__(self, *args):
        """Return True for any pair of users. """
        return True

class MinCommonFilter:
    
    """Pre-filtering method that filters out pairs of users that have less
    than a predefined value of common examples.
     
    """
    #NOTE: Not yet implemented.
    pass

