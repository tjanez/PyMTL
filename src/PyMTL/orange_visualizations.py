#
# orange_visualizations.py
# Methods for visualizing Orange classes (learners, classifiers, etc.).
#
# Copyright (C) 2010, 2013 Tadej Janez
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

import inspect, os, sys

import orange

# extend PYTHONPATH so that we can load Orange's widgets
orange_path = os.path.dirname(inspect.getfile(orange))
sys.path.extend([os.path.join(orange_path, "OrangeWidgets"),
                 os.path.join(orange_path, "OrangeWidgets/Classify")])
import OWDlgs
import OWClassificationTreeGraph


def save_treegraph_image(tree, save_path):
    """Render the tree and save it to the given file path.
    
    Note: Currently Qt only supports .svg and .png image formats.
    
    Arguments:
    tree -- Orange.classification.tree.TreeClassifier object representing the
        tree to be rendered
        NOTE: The tree learner had to have the attribute store_instances set to
        True.
    save_path -- string representing the path of the file, where to save the
        image of the rendered tree
    
    """
    # create the classification tree graph widget
    ow = OWClassificationTreeGraph.OWClassificationTreeGraph()
    # render the given tree
    ow.ctree(tree)
    # set node color to major class probability
    ow.NodeColorMethod = 2
    ow.toggleNodeColor()
    # create a 'Choose image size' dialog with the appropriate scene object
    sizeDlg = OWDlgs.OWChooseImageSizeDlg(ow.scene)
    # save the tree graph to a file
    sizeDlg.saveImage(save_path)

if __name__ == "__main__":
    # initialize Qt application
    from OWWidget import QApplication
    a = QApplication(sys.argv)
    
    # find out the current file's location so it can be used to compute the
    # location of other files/directories
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))

    import Orange
    data = Orange.data.Table(os.path.join(path_prefix,
                                          "data/demo/bool_func.tab"))
    ct = Orange.classification.tree.TreeLearner(data, store_instances=True)
    save_treegraph_image(ct, os.path.join(path_prefix, "results/test-tree.svg"))
    