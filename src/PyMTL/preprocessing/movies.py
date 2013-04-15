# -*- coding: utf-8 -*-
#
# movies.py
# Classes and methods for preprocessing the raw iTivi movie data.
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

import logging, re
from collections import OrderedDict

import Orange

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def is_series(attrs):
    """Return True if the given movie is a series, False otherwise.
    
    Arguments:
    attrs -- dictionary containing movie's attributes and their values
    
    """
    title = str.lower(attrs["Title"])
    if re.search(r"disk \d", title) or re.search(r"disc \d", title):
    	return True
    elif re.search(r"sezona", title):
        return True
    elif re.search(r"epizode", title):
        return True
    elif re.search(r"nekoč je bilo\s*\.\.\.\s*življenje", title):
        return True
    elif re.search(r"sveto pismo \d+", title):
        return True
    elif "tv-series" in attrs["Genre"]:
        return True
    else:
     	return False

def is_porno(attrs):
    """Return True if the given movie is a porno movie, False otherwise.
    
    Arguments:
    attrs -- dictionary containing movie's attributes and their values
    
    """
    if "adult" in attrs["Genre"]:
        return True
    else:
        return False

def is_book(attrs):
    """Return True if the given movie is a book, False otherwise.
    
    Arguments:
    attrs -- dictionary containing movie's attributes and their values
    
    """
    title = str.lower(attrs["Title"])
    if re.search(r"\(knjiga\)", title):
        return True
    elif re.search(r"^izvor vrst$", title):
        return True
    elif re.search(r"^sašo ožbolt: anti-politična", title):
        return True
    else:
        return False

def is_game(attrs):
    """Return True if the given movie is a video game, False otherwise.
    
    Arguments:
    attrs -- dictionary containing movie's attributes and their values
    
    """
    title = str.lower(attrs["Title"])
    if re.search(r"\(wii\)", title):
        return True
    else:
        return False

class RawDataPreprocessor:

    """A class for transforming the raw iTivi movie data to an Orange data
    table.
      
    """
    
    # dictionary mapping from attribute ids (as used in the iTivi database)
    # to their names (as used in the Orange domain)
    _ATTR_ID_NAME = {"1": "Genre",
                     "2": "Year",
                     "3": "Length",
                     "4": "Actor",
                     "5": "Director",
                     "6": "Voice-actor",
                     "7": "Title"}
    # list of attributes with one value
    _ATTR_ONE_VALUE = ["Year", "Length", "Director", "Title"]
    
    def __init__(self, raw_data_file):
        """Read the raw_data_file, parse it and convert each line to a triplet:
        (movie_id, attr_id, attr_value).
        Create a dictionary mapping from movies' ids to a dictionary mapping
        from movies' attributes to their values. Store this dictionary in the
        self._movie_attrs variable.
        
        Keyword arguments:
        raw_data_file -- string representing the path to the raw iTivi movie
            data
        
        """
        # read the file and store the triples in a list
        with open(raw_data_file) as raw_data:
            triplets = []
            for i, line in enumerate(raw_data):
                if i == 0:
                    # skip the first line as it contains attribute descriptions
                    continue
                line = line.rstrip()
                # split lines into 3 parts: movie id, attribute id,
                # attribute value
                movie_id, attr_id, attr_value = line.split(None, 2)
                triplets.append((movie_id, attr_id, attr_value))
        # create a double-layered dictionary with movie attributes
        self._movie_attrs = OrderedDict()
        for movie_id, attr_id, attr_value in triplets:
            if movie_id not in self._movie_attrs:
                self._movie_attrs[movie_id] = dict()
            attr_name = self._ATTR_ID_NAME[attr_id]
            if attr_name in self._ATTR_ONE_VALUE:
                self._movie_attrs[movie_id][attr_name] = attr_value
            elif attr_name not in self._movie_attrs[movie_id]:
                self._movie_attrs[movie_id][attr_name] = [attr_value]
            else:
                self._movie_attrs[movie_id][attr_name].append(attr_value)
    
    def purge_database(self):
        """Purge the movie database and remove all entries that are either:
        - series, or
        - porno movies, or
        - books, or
        - video games.
        Report on the number of deleted entries.
        
        """
        n = len(self._movie_attrs)
        # drop all series
        for movie_id, attrs in list(self._movie_attrs.iteritems()):
            if is_series(attrs):
                del self._movie_attrs[movie_id]
                logging.debug("Deleted movie (cause: series): {} (id: {})".\
                              format(attrs["Title"], movie_id))
        logging.info("Dropped {} series.".format(n - len(self._movie_attrs)))
        n = len(self._movie_attrs)
        # drop all porno movies
        for movie_id, attrs in list(self._movie_attrs.iteritems()):
            if is_porno(attrs):
                del self._movie_attrs[movie_id]
                logging.debug("Deleted movie (cause: porno): {} (id: {})".\
                              format(attrs["Title"], movie_id))
        logging.info("Dropped {} porno movies.".format(n - 
                                                       len(self._movie_attrs)))
        n = len(self._movie_attrs)
        # drop all books
        for movie_id, attrs in list(self._movie_attrs.iteritems()):
            if is_book(attrs):
                del self._movie_attrs[movie_id]
                logging.debug("Deleted movie (cause: book): {} (id: {})".\
                              format(attrs["Title"], movie_id))
        logging.info("Dropped {} books.".format(n - len(self._movie_attrs)))
        n = len(self._movie_attrs)
        # drop video games
        for movie_id, attrs in list(self._movie_attrs.iteritems()):
            if is_game(attrs):
                del self._movie_attrs[movie_id]
                logging.debug("Deleted movie (cause: game): {} (id: {})".\
                              format(attrs["Title"], movie_id))
        logging.info("Dropped {} games.".format(n - len(self._movie_attrs)))
        n = len(self._movie_attrs)
#        # DEBUG
#        # print remaining movies
#        for movie_id, attrs in self._movie_attrs.iteritems():
##        	print "{} (id: {})".format(attrs["Title"], movie_id)
#            print "{}".format(attrs["Title"])

    def _find_frequent_actors(self, k):
        """Select k most frequently appearing actors from the database.
        Store a list ordered in descending order in the
        self._frequent_actors variable.
    
        Keyword arguments:
        k -- integer representing the number of actors to select
            
        """
        if k < 0:
            raise ValueError("Number of actors to select should be > 0.")
        # count actor appearances
        actors = dict()
        for attrs in self._movie_attrs.itervalues():
            if "Actor" in attrs:
                for actor in attrs["Actor"]:
                    if actor in actors:
                        actors[actor] += 1
                    else:
                        actors[actor] = 1
        sorted_actors = sorted(actors.iteritems(), key=lambda x: x[1],
                               reverse=True)
        logging.info("Total number of actors: {}".format(len(sorted_actors)))
        self._frequent_actors = [actor for actor, _ in sorted_actors[:k]]
#        # DEBUG
#        for actor in self._frequent_actors:
#            print actor

    def _find_genres(self):
        """Search for all genres appearing in the database.
       Store a list ordered in descending order (of the most frequently
       appearing genres) in the self._genres variable.
        
        """
        genres = dict()
        # count genre appearances
        for attrs in self._movie_attrs.itervalues():
            if "Genre" in attrs:
                for genre in attrs["Genre"]:
                    if genre in genres:
                        genres[genre] += 1
                    else:
                        genres[genre] = 1
        sorted_genres = sorted(genres.iteritems(), key=lambda x: x[1],
                               reverse=True)
        logging.info("Total number of genres: {}".format(len(sorted_genres)))
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
        id_ = Orange.data.variable.String(name="Id")
        title = Orange.data.variable.String(name="Title")
        director = Orange.data.variable.String(name="Director")
        # create a new class-less domain
        domain = Orange.data.Domain([year, length] + freq_actors + \
                                    [n_freq_actors] + genres, False)
        # add meta attributes
        meta_attributes = {Orange.data.new_meta_id() : attr for attr in
                           [id_, title, director]}
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
        # iterate over the whole database
        for movie_id in self._movie_attrs:
            # create (or retrieve) an Orange instance corresponding to the movie
            if movie_id not in movies:
                ins = self._create_instance()
                ins["Id"] = movie_id
                movies[movie_id] = ins
            else:
                ins = movies[movie_id]
            for attr_name, attr_value in self._movie_attrs[movie_id].iteritems():
                # store the value if the attribute only has one value
                if attr_name in self._ATTR_ONE_VALUE:
                    ins[attr_name] = attr_value
                # store the genre value
                elif attr_name == "Genre":
                    for genre in attr_value:
                        ins[genre] = "yes"
                # store the actor value if the actor is among the most
                # frequently appearing actors
                elif attr_name == "Actor":
                    for actor in attr_value:
                        if actor in self._frequent_actors:
                            ins[actor] = "yes"
                # ignore the "voice-actors" in animated movies
                elif attr_name == "Voice-actor":
                    pass
                else:
                    raise ValueError("Unknown attribute id: {} with value: {}".\
                                     format(attr_id, attr_value))
        # compute the number of frequent actors appearing in each movie
        for movie_id, ins in movies.iteritems():
            s = 0
            for i in range(2, 2 + len(self._frequent_actors)):
                if ins[i] == "yes":
                    s += 1
            ins["# Freq Actors"] = s
        # sanity check
        for movie_id, ins in movies.iteritems():
            manually_checked_length_exceptions = ["504"]
            try:
                length = int(ins["Length"])
                if (not 24 < length < 250 and
                    movie_id not in manually_checked_length_exceptions):
                    logging.warning("'{}' has length {} minutes (id: {})".\
                                    format(ins["Title"], length, movie_id))
            except TypeError as e:
                logging.warning("'{}' has unknown length (id: {})".\
                                format(ins["Title"], movie_id))
            try:
                year = int(ins["Year"])
                if not 1910 < year < 2012:
                    logging.warning("'{}' has year {} (id: {})".\
                                    format(ins["Title"], year, movie_id))
            except TypeError as e:
                logging.warning("'{}' has unknown year (id: {})".\
                                format(ins["Title"], movie_id))
        
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
    preprocessor.purge_database()
    preprocessor.create_domain(128)
    preprocessor.create_datatable()
    preprocessor.save_datatable(os.path.join(path_prefix, "data/movies.tab"))
    