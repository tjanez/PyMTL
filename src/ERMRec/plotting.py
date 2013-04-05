#
# plotting.py
# Contains classes and methods for storing plot descriptions and drawing
# different plots showing the results of the learning algorithms.
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

from math import ceil

import numpy as np
# select a non-interactive backend for matplotlib to enable this script to be
# run without an open display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class BarPlotDesc:
    
    """A class containing a bar plot description."""
    
    def __init__(self, left_edges, heights, width, yerr, label, color="blue",
                 ecolor="red"):
        """Initialize a BarPlotDesc object. Store the given parameters as
        attributes.
        
        Arguments:
        left_edges -- list of bars' left edges
        heights -- list of bars' heights
        width -- float representing bars' widths
        yerr -- list of values representing the heights of the +/- error bars
        label -- string representing bar plot's label to be used in the legend
        
        Keyword arguments:
        color -- string representing the color of the bars
        ecolor -- string representing the color of error bars 
        
        """
        self.left_edges = left_edges
        self.heights = heights
        self.width = width
        self.yerr = yerr
        self.color = color
        self.ecolor = ecolor
        self.label = label
    
        
class LinePlotDesc:
    
    """A class containing a line plot description."""
    
    def __init__(self, x, y, yerr, label, color="blue", ecolor="red"):
        """Initialize a LinePlotDesc object. Store the given parameters as
        attributes.
        
        Arguments:
        x -- list of line's x points
        y -- list of line's y points
        yerr -- list of values representing the heights of the +/- error bars
        label -- string representing bar plot's label to be used in the legend
        
        Keyword arguments:
        color -- string representing the color of the bars
        ecolor -- string representing the color of error bars
        
        """
        self.x = x
        self.y = y
        self.yerr = yerr
        self.color = color
        self.ecolor = ecolor
        self.label = label

def _draw_subplot(axes, plot_descs, title="", xlabel="", ylabel=""):
    """Draw the given (bar or line) plots on the given Axes object.
    
    Arguments:
    axes -- matplotlib.axes.Axes object where to draw the plot
    plot_descs -- list of BarPlotDesc or LinePlotDescobjects, one for each
        learner
    
    Keyword arguments:
    title -- string representing plot's title
    xlabel -- string representing x axis's label
    ylabel -- string representing y axis's label
    
    """
    for pd in plot_descs:
        if isinstance(pd, BarPlotDesc):
            axes.bar(pd.left_edges, pd.heights, width=pd.width, yerr=pd.yerr,
                color=pd.color, ecolor=pd.ecolor, label=pd.label, alpha=0.75)
        elif isinstance(pd, LinePlotDesc):
            axes.errorbar(pd.x, pd.y, yerr=pd.yerr, color=pd.color,
                ecolor=pd.ecolor, label=pd.label, alpha=0.75)
        else:
            raise ValueError("Unsupported plot type: '{}'".format(pd.__class__\
                                                                  .__name__))
    axes.set_title(title, size="small")
    axes.set_xlabel(xlabel, size="small")
    axes.set_ylabel(ylabel, size="small")
    axes.set_xlim(left=0.0)
    axes.set_ylim(0.0, 1.0)
    axes.grid(b=True)
    axes.legend(loc="upper right", fancybox=True,
               prop=FontProperties(size="x-small"))

def plot_multiple(plot_descs_mult, file_name, title="", subplot_title_fmt="{}",
                  xlabel="", ylabel=""):
    """Plot multiple subplots on one figure.
    The method can draw from one up to eight subplots on a figure. It
    automatically arranges the subplots into the appropriate number of rows and
    columns.
    
    Arguments:
    plot_descs_mult -- ordered dictionary with items of the form (name,
        plot_descs), where name is a string representing the base learner's name
        and plot_descs is a list of BarPlotDesc or LinePlotDesc objects, one for
        each learner
    file_name -- string representing the path where to save the drawn figure
    
    Keyword arguments:
    title -- string representing the title of the whole plot
    subplot_title_fmt -- string representing a template for the subplot titles;
        it must contain exactly one pair of braces ({}), where the base
        learner's name will be put
    xlabel -- string representing subplots' x axis's labels
    ylabel -- string representing subplots' y axis's labels
     
    """
    # figure sizes in inches (width, height)
    a4 = (8.3, 11.7)
    a4_landscape = (11.7, 8.3)
    # create the appropriate number of subplots and set the appropriate figure
    # size
    nplots = len(plot_descs_mult)
    if nplots == 1:
        nrows, ncols = 1, 1
        figsize = a4_landscape
    elif nplots in (2, 3):
        nrows, ncols = nplots, 1
        figsize = a4
    elif nplots == 4:
        nrows, ncols = 2, 2
        figsize = a4_landscape
    elif nplots in (5, 6, 7, 8):
        nrows, ncols = ceil(1.*nplots / 2), 2
        figsize = a4
    else:
        raise ValueError("Too many subplots to draw!")
    fig = plt.figure(figsize=figsize)
    # create an empty object array to hold all axes; it's easiest to make it 1-d
    # so we can just append subplots upon creation
    axarr = np.empty(nplots, dtype=object)
    # Note: off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots+1):
        axarr[i-1] = fig.add_subplot(nrows, ncols, i)
    # draw plots to the subplots
    for i, (bl, plot_descs) in enumerate(plot_descs_mult.iteritems()):
        _draw_subplot(axarr[i], plot_descs, title=subplot_title_fmt.format(bl),
                  xlabel=xlabel, ylabel=ylabel)
    # adjust figure parameters to make it look nicer
    fig.suptitle(title)
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.10, right=0.95,
                        wspace=0.2, hspace=0.25)
    fig.savefig(file_name)
