#
# users.py
# Classes and methods for preprocessing the raw iTivi users' ratings.
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

import os, re

import Orange

from ERMRec.config import *

class RawDataPreprocessor:

    """A class for transforming the raw iTivi users' ratings to a series of
    Orange data tables.
      
    """
    
    def __init__(self, raw_data_file):
        """Read the raw_data_file, parse it and convert each line to a triplet:
        (movie_id, user_id, rating).
        Store the list of triplets in the self._database variable.
        
        """
        with open(raw_data_file) as raw_data:
            self._database = []
            for i, line in enumerate(raw_data):
                if i == 0:
                    # skip the first line as it contains attribute descriptions
                    continue
                line = line.rstrip()
                # split lines into 4 parts: movie id, user id, rating
                # timestamp (irrelevant), rating
                movie_id, user_id, _, rating = line.split('\t', 3)
                self._database.append((movie_id, user_id, rating))
    
    def _compute_binarized_ratings(self, table):
        """Compute the binarized ratings for the given Orange data table.
        Read numerical ratings from the "Rating (raw)" attribute and store
        the binarized ratings to the "Rating" attribute.
        Give the "like" value to instances which exceed the average rating (over
        all instances in the data table). Otherwise, give instances the
        "dislike" value.
        
        Keyword arguments:
        table -- Orange.data.Table with movie instances and ratings
        
        """
        avg_rat = 1.* sum([int(ins["Rating (raw)"]) for ins in table]) / \
                    len(table)
        for ins in table:
            if ins["Rating (raw)"] >= avg_rat:
                ins["Rating"] = "like"
            else:
                ins["Rating"] = "dislike"
    
    def create_datatables(self, m, movies_file):
        """Create Orange data tables for all users who have at least m ratings.
        Each data table will contain all the movies the user rated along with
        their binarized ratings (like/dislike).
        Store the created Orange.data.Table objects in the self._datatables
        dictionary with keys corresponding to users' ids.
        
        Keyword arguments:
        m -- integer representing the minimal value of ratings the user has to
            have in order to create a data table for him
        movies_file -- string representing the path to the movies data table 
        
        """
        movies = Orange.data.Table(movies_file)
        # create a new domain from the movies domain and add two new attributes:
        # numerical (raw) rating and binarized (like/dislike) rating
        rating = Orange.data.variable.Discrete(name="Rating",
                                               values=["like", "dislike"])
        raw_rating = Orange.data.variable.Continuous(name="Rating (raw)",
                                                     number_of_decimals=1)
        new_domain = Orange.data.Domain(movies.domain, rating)
        new_domain.add_metas(movies.domain.get_metas())
        new_domain.add_meta(Orange.data.new_meta_id(), raw_rating)
        
        users = dict()
        # iterate over the whole database
        for movie_id, user_id, rating in self._database:
            # create (or retrieve) an Orange data table corresponding to the
            # user
            if user_id not in users:
                table = Orange.data.Table(new_domain)
                users[user_id] = table
            else:
                table = users[user_id]
            ins = Orange.data.Instance(new_domain, movies[int(movie_id)-1])
            if ins["Id"] != movie_id:
                raise ValueError("Movie at position {} does not have id={}".\
                                 format(int(movie_id)-1, movie_id))
            ins["Rating (raw)"] = rating
            table.append(ins)
        logging.debug("Total number of users: {}".format(len(users)))
        # only keep users with at least m ratings
        users = {user_id : table for user_id, table in users.iteritems()
                 if  len(table) >= m}
        logging.debug("Kept {} users who have more than {} ratings".\
                      format(len(users), m))
        # compute binarized ratings for users
        for table in users.itervalues():
            self._compute_binarized_ratings(table)
        self._users_datatables = users
    
    def save_datatables(self, dir_path):
        """Store the Orange.data.Table objects in self._users_datatables to
        separate files in the directory with the given path.
        
        Keyword arguments:
        dir_path -- string representing the path of the directory where to
            store the users' data tables 
        
        """
        # create the directory if it doesn't exist or clean the files otherwise
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            for file in os.listdir(dir_path):
                if re.search(r"user\d+\.tab", file):
                    os.remove(os.path.join(dir_path, file))
        for user_id, table in self._users_datatables.iteritems():
            table.save(os.path.join(dir_path, "user{:0>5}.tab".format(user_id)))
    
if __name__ == "__main__":
    # compute the location of the raw data file from current file's location
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../../"))
    raw_data_file = os.path.join(path_prefix, "data/itivi_raw/ratings.csv")
    movies_file = os.path.join(path_prefix, "data/movies.tab")
    
    # process the raw data file
    preprocessor = RawDataPreprocessor(raw_data_file)
    preprocessor.create_datatables(10, movies_file)
    preprocessor.save_datatables(os.path.join(path_prefix, "data/users-m10"))
    