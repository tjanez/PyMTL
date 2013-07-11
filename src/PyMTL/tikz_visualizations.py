#
# tikz_visualizations.py
# Methods for visualizing PyMTL objects (learners, classifiers, etc.) with TikZ.
#
# Copyright (C) 2013 Tadej Janez
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

import re

from Orange.core import DefaultClassifier
from Orange.classification.tree import TreeClassifier

# indentation
IND = "  "


def _convert_attr_name(name):
    """Convert the given attribute name to a 'prettified' TeX version.
    
    Parameters
    ----------
    name : string
        The attribute name.
    
    Returns
    -------
    res : string
        The 'prettified' TeX version of the attribute name.
    
    """
    if name == "id":
        return "\\id"
    else:
        match = re.search(r"^x(\d+)$", name)
        if match:
            return "$x_{}$".format(match.group(1))
        else:
            raise ValueError("Unexpected attribute name: {}".format(name))


def _convert_id_branch_desc(branch_desc):
    """Convert the given branch description that describes the split on the 'id'
    attribute to a shortened and 'prettified' TeX version.
    
    Parameters
    ----------
    branch_desc : string
        A branch description that describes the split on the 'id' attribute.
        
    Returns
    -------
    res : (string, string)
        The shortened and 'prettified' TeX version of branch's description
        and its negation.
    
    """
    groups = re.findall(r"Group \d+, task \d+", branch_desc)
    conv_groups = []
    for g in groups:
        match = re.search(r"Group (\d+), task (\d+)", g)
        g_number, t_number = match.group(1), match.group(2)
        conv_groups.append("G_{}T_{}".format(g_number, t_number))
    return ("$ \\in \{" + ", ".join(conv_groups) + "\}$",
            "$ \\notin \{" + ", ".join(conv_groups) + "\}$")


def _id_branch_desc_size(branch_desc):
    """Return the size of the branch description that describes the split on the
    'id' attribute in terms of the number different values it has.
    
    Parameters
    ----------
    branch_desc : string
        A branch description that describes the split on the 'id' attribute.
    
    Returns
    -------
    res : int
        The number of different values of the 'id' attribute in branch's
        description.
    
    """
    return len(re.findall(r"Group \d+, task \d+", branch_desc))


def _draw_tikz_subtree(node, ind_level, only_draw=None):
    """Draw a tikz-qtree-formatted subtree starting at the given node.
    
    Parameters
    ----------
    node : Orange's TreeNode
        The root node of a subtree for which to draw a tikz-qtree-formatted
        tree.
    ind_level : int
        The starting indentation level.
    only_draw : string (optional)
        If given, the method will only draw the subtree while the current split
        attribute is equal to the only_draw value.
    
    Returns
    -------
    res : string
        The string representation of the tikz-qtree-formatted subtree starting
        at the given node.
    
    """
    res = ""
    if node.branches == None:
        if isinstance(node.node_classifier, DefaultClassifier):
            res += (ind_level * IND + "{}\n".
                    format(node.node_classifier.default_value))
        else:
            raise ValueError("Currently, only DefaultClassifier is supported"
                             " as the leaf classifier.")
    elif len(node.branches) == 2:
        split_attr = node.branch_selector.class_var.name
        conv_split_name = _convert_attr_name(split_attr) 
        if only_draw == None or split_attr == only_draw:
            # recursively draw the subtrees
            res += ind_level * IND + "[.{{{}}}\n".format(conv_split_name)
            branch0_size = _id_branch_desc_size(node.branch_descriptions[0])
            branch1_size = _id_branch_desc_size(node.branch_descriptions[1])
            if branch0_size < branch1_size:
                # use left branch's description and its negation because it is
                # shorter
                conv_branch_desc_pair = _convert_id_branch_desc(
                                            node.branch_descriptions[0])
                lbranch = node.branches[0]
                rbranch = node.branches[1]
            elif branch1_size < branch0_size:
                # use right branch's description and its negation because it is
                # shorter
                conv_branch_desc_pair = _convert_id_branch_desc(
                                            node.branch_descriptions[1])
                lbranch = node.branches[1]
                rbranch = node.branches[0]
            else:
                # combine the converted versions of branches' descriptions
                # because they have the same length
                conv_branch_desc_pair = \
                    (_convert_id_branch_desc(node.branch_descriptions[0])[0],
                     _convert_id_branch_desc(node.branch_descriptions[1])[0])
                lbranch = node.branches[0]
                rbranch = node.branches[1]
            res += ((ind_level + 1) * IND + "\\edge node[auto=right]{{{}}};\n".
                    format(conv_branch_desc_pair[0]))
            res += _draw_tikz_subtree(lbranch, ind_level + 1,
                                      only_draw=only_draw)
            res += ((ind_level + 1) * IND + "\\edge node[auto=left]{{{}}};\n".
                    format(conv_branch_desc_pair[1]))
            res += _draw_tikz_subtree(rbranch, ind_level + 1,
                                      only_draw=only_draw)
            res += ind_level * IND + "]\n"
        else:
            # stop drawing subtrees and indicate that the tree was cut by
            # drawing '\ldots'
            res += ind_level * IND + "[.{{{}}}\n".format(conv_split_name)
            for dir, branch, branch_desc in zip(("right", "left"),
                                                node.branches,
                                                node.branch_descriptions):
                res += ((ind_level + 1) * IND + "\\edge node[auto={}]{{{}}};\n".
                        format(dir, branch_desc))
                res += (ind_level + 1) * IND + "$\\ldots$\n"
            res += ind_level * IND + "]\n"
    else:
        raise ValueError("Currently, only binary trees are supported.")
    return res


def draw_tikz_tree(tree):
    """Draw a tikz-qtree-formatted tree from the given Orange TreeClassifier.
    
    Parameters
    ----------
    tree : Orange's TreeClassifier
        The decision tree for which to draw a tikz-qtree-formatted tree.
    
    Returns
    -------
    res : string
        The string representation of the tikz-qtree-formatted tree.
    
    """
    if not isinstance(tree, TreeClassifier):
        raise ValueError("Unsupported decision tree type: {}".
                         format(type(tree)))
    initial_ind_level = 3
    node = tree.tree
    res = _draw_tikz_subtree(node, initial_ind_level, only_draw="id")
    # insert the '\Tree' command at the beginning of the string
    res = "\\Tree " + res[6:]
    # indicate that the following is a TikZ picture
    res = "\\begin{tikzpicture}\n" + res
    res += "\\end{tikzpicture}\n"
    return res


if __name__ == "__main__":
    import os.path
    # find out the current file's location so it can be used to compute the
    # location of other files/directories
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.abspath(os.path.join(cur_dir, "../../"))
    
    results_path = os.path.join(path_prefix, "results/binarization_experiment/")
    
    import Orange.classification.tree as octree
    import Orange.feature.scoring as fscoring
    split_const = octree.SplitConstructor_ExhaustiveBinary(measure=
                                                           fscoring.InfoGain())
    tree_learner = octree.TreeLearner(split=split_const, min_instances=10,
                            same_majority_pruning=True, store_instances=True)
    from Orange.data import Table
    
    # TEST for equality of "original" vs. "pickled/unpickled" Orange trees
    from PyMTL.util import pickle_obj, unpickle_obj
    import numpy as np
    for i in range(10):
        data = Table(os.path.join(results_path, "bool_func-a8d4n100g2tg5nse0.0rs15"
                              "nls10-seed63-complete_test/orange_merged_learn-"
                              "repetition{}.tab".format(i)))
        tree = tree_learner(data)
        pickle_path = os.path.join(results_path, "test-pickle.pkl")
        pickle_obj(tree, pickle_path)
        unpickled_tree = unpickle_obj(pickle_path)
        print ("Repetition {} original vs. pickled/unpickled tree equality:".
               format(i)),
        print np.all(tree[e] == unpickled_tree[e] for e in data)
    os.remove(pickle_path)
    
    data = Table(os.path.join(results_path, "bool_func-a8d4n100g2tg5nse0.0rs15"
                              "nls10-seed63-complete_test/orange_merged_learn-"
                              "repetition0.tab"))
    tree = tree_learner(data)
    print draw_tikz_tree(tree)
    
    # initialize Qt application
    import sys
    from OWWidget import QApplication
    a = QApplication(sys.argv)
    from PyMTL.orange_visualizations import save_treegraph_image
    save_treegraph_image(tree, os.path.join(results_path, "test-tree.svg"))
    