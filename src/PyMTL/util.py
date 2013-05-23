#
# util.py
# Contains auxiliary classes and methods for the PyMTL package. 
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

import logging

def configure_logger(logger, level=logging.DEBUG, console_level=logging.DEBUG,
                     file_name=None, file_level=logging.DEBUG):
    """Configure the given Logger instance.
    
    Keyword arguments:
    logger -- logging.Logger object
    level -- level of the created logger
    console_level -- level of the console handler attached to the created logger
    file_name -- file name of the file handler attached to the created logger;
        if None, no file handler is created 
    file_level -- level of the file handler attached to the created logger
    
    """
    # remove previous handler of the given Logger object
    _remove_handlers(logger)
    # set logging level
    logger.setLevel(level)
    # create formatter
    formatter = logging.Formatter(fmt="[%(asctime)s] %(name)-15s "
                    "%(levelname)-7s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # create console handler and configure it
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create a file handler and set its level
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def _remove_handlers(logger):
    """Remove all handlers of the given Logger object.
    
    Arguments:
    logger -- logging.Logger object
    
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

logger_name = "PyMTL"
logger = logging.getLogger(logger_name)
configure_logger(logger)
