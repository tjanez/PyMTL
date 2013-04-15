#
# users.py
# Classes and methods for preprocessing the raw iTivi users' ratings.
#
# Copyright (C) 2012, 2013 Tadej Janez
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

import logging, math, os, re

# select a non-interactive backend for matplotlib to enable this script to be
# run without an open display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Orange

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _all_ratings_same(table):
    """Check if the ratings in the given Orange data table are all the same.
    Read the numerical ratings from the "Rating (raw)" attribute.
    Return True if all numerical ratings are the same, and False otherwise
    
    Keyword arguments:
    table -- Orange.data.Table with movie instances and ratings
      
    """
    for i in range(len(table) - 1):
        if table[i+1]["Rating (raw)"] != table[i]["Rating (raw)"]:
            return False
    return True

def _compute_binarized_ratings(table):
    """Compute the binarized ratings for the given Orange data table.
    Read numerical ratings from the "Rating (raw)" attribute and store
    the binarized ratings to the "Rating" attribute.
    Give the "like" value to instances which exceed (or are equal to) the
    average rating (over all instances in the data table). Otherwise, give
    instances the "dislike" value.
    
    Keyword arguments:
    table -- Orange.data.Table with movie instances and ratings
    
    """
    avg_rat = 1.* sum([int(ins["Rating (raw)"]) for ins in table]) / len(table)
    for ins in table:
        if ins["Rating (raw)"] >= avg_rat:
            ins["Rating"] = "like"
        else:
            ins["Rating"] = "dislike"

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
    
    def create_datatables(self, movies_file, selection_type="min_ratings",
                          p=10):
        """Create Orange data tables for according to the specified
        selection_type and the corresponding value of parameter p.
        Each data table will contain all the movies the user rated along with
        their binarized ratings (like/dislike).
        Store the created Orange.data.Table objects in the self._datatables
        dictionary with keys corresponding to users' ids.
        
        Arguments:
        movies_file -- string representing the path to the movies data table
         
        Keyword arguments:
        selection_type -- string (optional) which indicates which criterion to
            use when selecting users:
            - "min_ratings" (default) -- users who have at least p ratings will
                be kept
            - "largest_users" -- p users with the maximal number of ratings will
                be kept
        p -- integer (optional) representing the p parameter for the
            selection_type
        
        """
        if selection_type not in ["min_ratings", "largest_users"]:
            raise ValueError("Unsupported selection_type: {}".\
                             format(selection_type))
        # store selection_type and p for later
        self._selection_type = selection_type
        self._p = p
        
        movies_table = Orange.data.Table(movies_file)
        # create a dictionary mapping from movie id to its Orange instance
        movies = {str(ins["Id"]) : ins for ins in movies_table}
        # create a new domain from the movies domain and add two new attributes:
        # numerical (raw) rating and binarized (like/dislike) rating
        rating = Orange.data.variable.Discrete(name="Rating",
                                               values=["like", "dislike"])
        raw_rating = Orange.data.variable.Continuous(name="Rating (raw)",
                                                     number_of_decimals=1)
        new_domain = Orange.data.Domain(movies_table.domain, rating)
        new_domain.add_metas(movies_table.domain.get_metas())
        new_domain.add_meta(Orange.data.new_meta_id(), raw_rating)
        
        skip = 0
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
            # retrieve the Orange instance corresponding to this movie id if it
            # exits
            if movie_id in movies:
                ins = Orange.data.Instance(new_domain, movies[movie_id])
                ins["Rating (raw)"] = rating
                table.append(ins)
            else:
                skip += 1
        logging.info("Skipped {} ratings since the corresponding movie was not "
                     "found in the movies data table.".format(skip))
        logging.info("Total number of users: {}".format(len(users)))
        # discard users with all the same ratings
        len_before = len(users)
        users = {user_id : table for user_id, table in users.iteritems()
                 if not _all_ratings_same(table)}
        logging.info("Discared {} users with ratings that are all the same".\
                      format(len_before - len(users)))
        # select users according to selection_type
        if selection_type == "min_ratings":
            # only keep users with at least p ratings
            users = {user_id : table for user_id, table in users.iteritems()
                     if len(table) >= p}
            logging.info("Kept {} users who have more than {} ratings".\
                         format(len(users), p))
        elif selection_type == "largest_users":
            # keep p users with the most number of ratings
            users = dict(sorted(users.iteritems(), key=lambda u: len(u[1]),
                                reverse=True)[:p])
            logging.info("Kept {} users with the most number of ratings.".\
                         format(len(users)))
        # compute binarized ratings for users
        for table in users.itervalues():
            _compute_binarized_ratings(table)
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
                if re.search(r"^user\d+\.tab$", file):
                    os.remove(os.path.join(dir_path, file))
        for user_id, table in self._users_datatables.iteritems():
            table.save(os.path.join(dir_path, "user{:0>5}.tab".format(user_id)))
    
    def plot_histogram(self, save_path):
        """Compute and plot a histogram for the number of ratings of each user.
        Save the histogram to the given file path.
        
        Keyword arguments:
        save_path -- string representing the path of the file where to save the
            histogram
         
        """
        n_instances = [len(table) for table in
                       self._users_datatables.itervalues()]
        n, bars, _ = plt.hist(n_instances, bins=50, facecolor="green",
                              alpha=0.75)
        # save the histogram data to a file for latter inspection 
        with open(save_path+".data", 'w') as hist_data:
            hist_data.write("# of ratings:   # of users:\n")
            for n_rat, n_users in zip(bars, n):
                hist_data.write("{: >13.1f}   {: >11}\n".format(n_rat, n_users))
        y_max = math.ceil(1.2*max(n))
        plt.xlabel("Number of ratings")
        plt.ylabel("Number of users")
        if self._selection_type == "min_ratings":
            plt.title("Histogram for users with at least {} ratings".\
                      format(self._p))
        elif self._selection_type == "largest_users":
            plt.title("Histogram for {} users with the most number of ratings".\
                      format(self._p))
        else:
            raise ValueError("Unsupported selection_type: {}".\
                             format(self._selection_type))
        plt.ylim(0, y_max)
        plt.grid(True)
        plt.savefig(save_path)
    
if __name__ == "__main__":
    # compute the location of the raw data file from current file's location
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../../"))
    raw_data_file = os.path.join(path_prefix, "data/itivi_raw/ratings.csv")
    movies_file = os.path.join(path_prefix, "data/movies.tab")
    
    preprocessor = RawDataPreprocessor(raw_data_file)
#    min_ins = 10
#    save_dir = os.path.join(path_prefix, "data/users-min{}".format(min_ins))
#    preprocessor.create_datatables(movies_file, selection_type="min_ratings",
#                                   p = min_ins)
    larg_users = 100
    save_dir = os.path.join(path_prefix, "data/users-larg{}".format(larg_users))
    preprocessor.create_datatables(movies_file, selection_type="largest_users",
                                   p = larg_users)
    preprocessor.save_datatables(save_dir)
    preprocessor.plot_histogram(os.path.join(save_dir, "histogram.png"))
