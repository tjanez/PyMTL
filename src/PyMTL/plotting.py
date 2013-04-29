#
# plotting.py
# Contains classes and methods for storing plot descriptions and drawing
# different plots showing the results of the learning algorithms.
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

def _draw_subplot(axes, plot_descs, title="", xlabel="", ylabel="",
                  x_tick_points=None, x_tick_labels=None):
    """Draw the given (bar or line) plots on the given Axes object.
    
    Arguments:
    axes -- matplotlib.axes.Axes object where to draw the plot
    plot_descs -- list of BarPlotDesc or LinePlotDesc objects, one for each
        learner
    
    Keyword arguments:
    title -- string representing plot's title
    xlabel -- string representing x axis's label
    ylabel -- string representing y axis's label
    x_tick_points -- list of floats (or integers) representing x axis tick
        positions
    x_tick_labels -- list of strings representing x axis tick labels
    
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
    # set x-axis' left limit a bit to the left and x-axis' right limit a bit to
    # the right so that the resulting plot looks nicer
    margin = (x_tick_points[1] - x_tick_points[0]) / 2.0
    axes.set_xlim(left=x_tick_points[0] - margin,
                  right=x_tick_points[-1] + margin)
    if x_tick_points != None and x_tick_labels != None:
        axes.set_xticks(x_tick_points)
        axes.set_xticklabels(x_tick_labels)
    axes.set_ylim(0.0, 1.0)
    axes.grid(b=True)
    axes.legend(loc="lower right", fancybox=True,
               prop=FontProperties(size="x-small"))

def plot_multiple(plot_descs_mult, file_name, title="", subplot_title_fmt="{}",
                  xlabel="", ylabel="",
                  x_tick_points=None, x_tick_labels=None):
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
    x_tick_points -- list of floats (or integers) representing subplots' x axis
        tick positions
    x_tick_labels -- list of strings representing subplots' x axis tick labels
     
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
                  xlabel=xlabel, ylabel=ylabel, x_tick_points=x_tick_points,
                  x_tick_labels=x_tick_labels)
    fig.suptitle(title)
    # adjust figure parameters to make it look nicer
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.10, right=0.95,
                        wspace=0.2, hspace=0.25)
    fig.savefig(file_name)

from scipy.cluster.hierarchy import dendrogram

def _draw_dendrogram(axes, Z, labels=None):
    """Draw the given linkage information as a dendrogram on the given Axes
    object. Change the drawing parameters so that the dendrogram will blend
    nicely into the figure showing multiple dendrograms.
    
    Arguments:
    axes -- matplotlib.axes.Axes object where to draw the plot
    Z -- numpy.ndarray in the format as specified in the
        scipy.cluster.hierarchy.linkage's docstring
    
    Keyword arguments:
    labels --  list or tuple (optional) where i-th value is the text to put
        under the i-th leaf node
    
    """
    # set current axes instance
    plt.sca(axes)
    # draw the dendrogram
    dendrogram(Z, labels=labels, orientation="left")
    # remove x-axis labels
    axes.set_xticks(())
    # remove the black border around axes
    for spine in axes.spines.itervalues():
        spine.set_visible(False)
    # decrease the font size of y tick labels
    for ytl in axes.get_yticklabels():
        ytl.set_fontsize("small")

def plot_dendrograms(dend_info, file_name, title="", ylabel=""):
    """Plot multiple dendrograms on one figure.
    The method stacks multiple dendrograms vertically and by removing subplots'
    x-axis labels and black borders around axes creates an impression that the
    whole figure is one big dendrogram.
    
    Arguments:
    dend_info -- list of tuples (Z, labels), where:
        Z -- numpy.ndarray in the format as specified in the
            scipy.cluster.hierarchy.linkage's docstring
        labels -- list of labels representing ids corresponding to each
            consecutive integer
    file_name -- string representing the path where to save the drawn figure
    
    Keyword arguments:
    title -- string representing the title of the whole plot
    ylabel -- string representing the figure's y axis label
    
    """
    nplots = len(dend_info)
    nrows = nplots
    ncols = 1
    # make figure the size of an A4 page
    fig = plt.figure(figsize=(8.3, 11.7))
    # create an empty object array to hold all axes; it's easiest to make it 1-d
    # so we can just append subplots upon creation
    axarr = np.empty(nplots, dtype=object)
    # Note: off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots+1):
        axarr[i-1] = fig.add_subplot(nrows, ncols, i)
    # draw dendrograms to the subplots
    for i, (Z, labels) in enumerate(dend_info):
        _draw_dendrogram(axarr[i], Z, labels)
    fig.suptitle(title)
    # draw a common y-axis label
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.05, 0.5, ylabel, rotation="vertical",
            horizontalalignment="center", verticalalignment="center")
    # adjust figure parameters to make it look nicer
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.20, right=0.97,
                        wspace=0.1, hspace=0.05)
    fig.savefig(file_name)

if __name__ == "__main__":
    # A simple example with two linkages transformed in a figure with two
    # dendrograms
    Z = np.array([[0., 1., 1., 2.],
                  [2., 3., 2., 2.],
                  [4., 5., 3., 4.]])
    labels = ['00009', '00038', '00016', '00033']
    Z1 = np.array([[0., 1., 1., 2.],
                   [2., 4., 2., 3.],
                   [3., 5., 3., 4.]])
    labels1 = ['00009', '00038', '00016', '00033']
    dend_info = [(Z, labels), (Z1, labels1)]
    # find out the current file's location so it can be used to compute the
    # location of other files/directories
    import os.path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    file_name = os.path.join(path_prefix, "results/temp-dend.png")
    plot_dendrograms(dend_info, file_name, title="Here comes the title",
                     ylabel="y-label")
    