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


import cPickle as pickle

def pickle_obj(obj, file_path):
    """Pickle the given object to the given file_path.
    
    Keyword arguments:
    file_path -- string representing the path to the file where to pickle
        the object
    
    """
    with open(file_path, "wb") as pkl_file:
        pickle.dump(obj, pkl_file, pickle.HIGHEST_PROTOCOL)


def unpickle_obj(file_path):
    """Unpickle an object from the given file_path.
    Return the reference to the unpickled object.
    
    Keyword arguments:
    file_path -- string representing the path to the file where the object is
        pickled
    
    """
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)


import os, subprocess

def convert_svgs_to_pdfs(path):
    """Find the SVG files in the given path, convert them to PDF, crop them and
    finally, remove them.
    
    Note: This function requires 'rsvg-convert' and 'pdfcrop' commands to be
    installed somewhere in $PATH.
    
    Arguments:
    path -- string representing the path to the directory where to convert the
        SVG files to PDF
    
    """
    for file in sorted(os.listdir(path)):
        base, ext = os.path.splitext(file)
        if ext.lower() == ".svg":
            svg_file = os.path.join(path, file)
            pdf_file = os.path.join(path, base + ".pdf")
            retcode = subprocess.call(["-c", "rsvg-convert --format pdf "
                        "--width 10000 --keep-aspect-ratio {} | pdfcrop - {}".\
                        format(svg_file, pdf_file)], shell=True)
            # remove the original file only if the conversion was successful
            if retcode == 0:
                os.remove(svg_file)


def build_and_crop_tex_files(path):
    """Find all TeX files in the given path, build them and crop the resulting
    PDFs.
    
    Note: This function requires 'pdflatex' and 'pdfcrop' commands to be
    installed somewhere in $PATH.
    
    Arguments:
    path -- string representing the path to the directory where to build and
        crop the source TeX files
    
    """
    for file in sorted(os.listdir(path)):
        base, ext = os.path.splitext(file)
        if ext.lower() == ".tex":
            tex_file = file
            pdf_file = os.path.join(path, base + ".pdf")
            subprocess.call(["-c", "pdflatex -interaction=batchmode {0} && "
                             "rm {1}.{{aux,log}} && "
                             "pdfcrop --margins 10 {1}.pdf {1}.pdf".
                             format(file, base)],
                            shell=True, cwd=path)


import warnings

def ignore_deprecation_warnings(func):
    """This is a decorator which can be used to ignore deprecation warnings
    occurring in a function.
    
    """
    def new_func(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func
