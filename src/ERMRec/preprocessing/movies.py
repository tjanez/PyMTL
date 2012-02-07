#
# movies.py
# Classes and methods for preprocessing the raw iTivi movie data.
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

from ERMRec.config import * 

class RawDataPreprocessor:

    """A class for transforming the raw iTivi movie data to an Orange Table.
      
    """
    
    # dictionary mapping from attribute ids (as used in the iTivi database)
    # to their names (as used in the Orange domain)
    _ATTR_ID_NAME = {"2": "Year",
                     "3": "Length",
                     "5": "Director",
                     "7": "Title"}
    
    def __init__(self, raw_data_file):
        """Read the raw_data_file, parse it and convert each line to a triplet:
        (movie_id, attr_id, attr_value).
        Store the list of triplets in the self._database variable.
        
        """
        with open(raw_data_file) as raw_data:
            self._database = []
            for i, line in enumerate(raw_data):
                if i == 0:
                    # skip the first line as it contains attribute information
                    continue
                line = line.rstrip()
                # split lines into 3 parts: movie id, attribute id,
                # attribute value
                movie_id, attr_id, attr_value = line.split(None, 2)
                self._database.append((movie_id, attr_id, attr_value))

    def _find_frequent_actors(self, k):
        """Select k most frequently appearing actors from the database.
        Store a list ordered in descending order in the
        self._frequent_actors variable.
    
        Keyword arguments:
        k -- integer representing the number of actors to select
            
        """
        if k < 0:
            raise ValueError("Number of actors to select should be > 0.")
        actors = dict()
        for _, attr_id, attr_value in self._database:
            # lines with attr_id = 4 describe actors' names
            if attr_id == "4":
                actor = attr_value
                if actor in actors:
                    actors[actor] += 1
                else:
                    actors[actor] = 1
        sorted_actors = sorted(actors.iteritems(), key=lambda x: x[1],
                               reverse=True)
        logging.debug("Total number of actors: {}".format(len(sorted_actors)))
        self._frequent_actors = [actor for actor, _ in sorted_actors[:k]]

    def _find_genres(self):
        """Search for all genres appearing in the database.
       Store a list ordered in descending order (of the most frequently
       appearing genres) in the self._genres variable.
        
        """
        genres = dict()
        for _, attr_id, attr_value in self._database:
            # lines with attr_id = 1 describe genres' names
            if attr_id == "1":
                genre = attr_value
                if genre in genres:
                    genres[genre] += 1
                else:
                    genres[genre] = 1
        sorted_genres = sorted(genres.iteritems(), key=lambda x: x[1],
                               reverse=True)
        logging.debug("Total number of genres: {}".format(len(sorted_genres)))
        self._genres = [genre for genre, _ in sorted_genres]

    def create_domain(self, k):
        """Create an Orange domain that contains the appropriate attributes for
        the movies domain.
        Store the created Orange.data.Domain in the self._domain variable.
        
        Keyword arguments:
        k -- integer representing the number of most frequently appearing actors
            to select
        
        """
        year = Orange.data.variable.Continuous(name="Year",
                                               number_of_decimals=0)
        length = Orange.data.variable.Continuous(name="Length",
                                                 number_of_decimals=0)
        # cast (k most frequent actors)
        self._find_frequent_actors(k)
        freq_actors = [Orange.data.variable.Discrete(name=act,
                        values=["yes", "no"]) for act in self._frequent_actors]
        # number of frequent actors
        n_freq_actors = Orange.data.variable.Continuous(name="# Freq Actors",
                                                        number_of_decimals=0)
        # genre
        self._find_genres()
        genres = [Orange.data.variable.Discrete(name=genre,
                    values=["yes", "no"]) for genre in self._genres]
        # meta attributes
        id = Orange.data.variable.String(name="Id")
        title = Orange.data.variable.String(name="Title")
        director = Orange.data.variable.String(name="Director")
        # create a new class-less domain
        domain = Orange.data.Domain([year, length] + freq_actors + \
                                    [n_freq_actors] + genres, False)
        # add meta attributes
        meta_attributes = {Orange.data.new_meta_id() : attr for attr in
                           [id, title, director]}
        domain.add_metas(meta_attributes)
        self._domain = domain

    def _create_instance(self):
        """Create an Orange instance of the movies domain and fill it with "no"
        values for frequent actors and genres.
        Return the created Orange.data.Instance.  
        
        """
        ins = Orange.data.Instance(self._domain)
        actors_start = 2
        actors_end = actors_start + len(self._frequent_actors)
        genres_start = actors_end + 1
        genres_end = genres_start + len(self._genres)
        for i in range(actors_start, actors_end) + \
            range(genres_start, genres_end):
            ins[i] = "no"
        return ins

    def create_datatable(self):
        """Create an Orange data table for the movies domain using self._domain
        as the domain.
        Store the created Orange.data.Table in the self._datatable variable.
         
        """
        movies = dict()
        # iterate over all whole database
        for movie_id, attr_id, attr_value in self._database:
            # create (or retrieve) an Orange instance corresponding to the movie
            if movie_id not in movies:
                ins = self._create_instance()
                ins["Id"] = movie_id
                movies[movie_id] = ins
            else:
                ins = movies[movie_id]
            # store the value if the current attribute is in self._ATTR_ID_NAME
            if attr_id in self._ATTR_ID_NAME:
                ins[self._ATTR_ID_NAME[attr_id]] = attr_value
            # store the genre value
            elif attr_id == "1":
                ins[attr_value] = "yes"
            # store the actor value if the actor is among the most frequently
            # appearing actors
            elif attr_id == "4":
                if attr_value in self._frequent_actors:
                    ins[attr_value] = "yes"
            # ignore the "voice-actors" in animated movies
            elif attr_id == "6":
                pass
            else:
                raise ValueError("Unknown attribute id: {} with value: {}".\
                                 format(attr_id, attr_value))
        # compute the number of frequent actors appearing in each movie
        for movie_id, ins in movies.iteritems():
            sum = 0
            for i in range(2, 2 + len(self._frequent_actors)):
                if ins[i] == "yes":
                    sum += 1
            ins["# Freq Actors"] = sum
        sorted_movies = sorted(movies.iteritems(), key=lambda x: int(x[0]))
        sorted_movies = [movie for _, movie in sorted_movies]
        logging.debug("Total number of movies: {}".format(len(sorted_movies)))
        self._datatable = Orange.data.Table(sorted_movies)
    
    def save_datatable(self, datatable_file):
        """Store the Orange.data.Table in self._datatable to the given file
        name.
        
        Keyword arguments:
        datatable_file -- string representing the path where to store the
            data table in self._datatable
        
        """
        self._datatable.save(datatable_file)

if __name__ == "__main__":
    # compute the location of the raw data file from current file's location
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../../"))
    raw_data_file = os.path.join(path_prefix, "data/itivi_raw/attributes.csv")
    
    # process the raw data file
    preprocessor = RawDataPreprocessor(raw_data_file)
    preprocessor.create_domain(128)
    preprocessor.create_datatable()
    preprocessor.save_datatable(os.path.join(path_prefix, "data/movies.tab"))
    