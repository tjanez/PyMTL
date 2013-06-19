#
# orange_utils.py
# Contains custom classes and wrappers for the Orange package.
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

import Orange
import Orange.classification.tree as octree
import Orange.feature.scoring as fscoring

class ForcedFirstSplitTreeLearner(octree.TreeLearner):
    
    """A custom version of Orange.classification.tree.TreeLearner which "forces"
    the first split to the specified attribute.
    
    """
        
    def __init__(self, first_split_attr=None, **kwargs):
        """Initialize the ForcedFirstSplitTreeLearner.
        
        Keyword arguments:
        first_splitt_attr -- string representing the name of the attribute to
            be used for the first split
        kwargs -- dictionary of keyword arguments passed to the standard
            Orange.classification.tree.TreeLearner
        
        """
        if first_split_attr == None:
            raise ValueError("Please, specify the 'first_split_attr'.")
        self.first_split_attr = first_split_attr
        super(ForcedFirstSplitTreeLearner, self).__init__(**kwargs)
    
    def _new_tree_node(self, instances):
        """Create a new Orange.classification.tree.Node object for the given
        instances. Compute the node's contingency table, distribution and
        classifier. Return the constructed Orange.classification.tree.Node
        object. 
        
        Arguments:
        instances -- Orange.data.Table holding instances corresponding to this
            tree node
        
        """
        node = octree.Node()
        node.examples = instances
        node.contingency = Orange.statistics.contingency.Domain(instances)
        node.distribution = node.contingency.classes
        if self.base_learner.node_learner != None:
            node_learner = self.base_leaner.node_learner
        else:
            node_learner = Orange.classification.majority.MajorityLearner() 
        node.node_classifier = node_learner(instances)
        return node
    
    def __call__(self, instances, weight=0):
        """Build a decision tree for the given instances according to the
        specified parameters.
        Return an Orange.classification.tree.TreeClassfier object with the
        constructed tree.
        
        Arguments:
        instances -- Orange.data.Table holding learning instances
        
        Keyword arguments:
        weight -- meta attribute with weights of instances (optional)
        
        """
        # create an (internal) Orange.core.TreeLearner object
        bl = self._base_learner()
        self.base_learner = bl

        # set the scoring criteria if it was not set by the user
        if not self._handset_split and not self.measure:
            if instances.domain.class_var.var_type == Orange.data.Type.Discrete:
                measure = fscoring.GainRatio()
            else:
                measure = fscoring.MSE()
            bl.split.continuous_split_constructor.measure = measure
            bl.split.discrete_split_constructor.measure = measure
        # set the splitter if it was set by the user
        if self.splitter != None:
            bl.example_splitter = self.splitter
        
        # set up a boolean list with one entry for each feature and select the
        # (single) feature that the SplitConstructor should consider
        candidate_feat = [feat.name == self.first_split_attr for feat in
                          instances.domain]
        # create the tree's root node
        root_node = self._new_tree_node(instances)
        # call the SplitConstructor for the root node manually
        bs, bd, ss, quality, spent_feature = self.split(instances, weight,
            root_node.contingency, root_node.distribution, candidate_feat,
            root_node.node_classifier)
        root_node.branch_selector = bs
        root_node.branch_descriptions = bd
        root_node.branch_sizes = ss
        # split the examples into subsets by calling the appropriate Splitter
        if self.splitter != None:
            splitter = self.splitter
        else:
            splitter = octree.Splitter_IgnoreUnknowns()
        subsets = splitter(root_node, root_node.examples)[0]
        # build a sub-tree for each subset (which is not None) and store it as
        # a branch of the root_node
        root_node.branches = []
        for subset in subsets:
            if subset != None:
                subtree = bl(subset, weight)
                root_node.branches.append(subtree.tree)
        # create an (internal) Orange.core.TreeClassifier object
        descender = getattr(self, "descender", octree.Descender_UnknownMergeAsBranchSizes())
        tree = octree._TreeClassifier(domain=instances.domain, tree=root_node,
                                      descender=descender)
        
        # perform post pruning
        if getattr(self, "same_majority_pruning", 0):
            tree = Pruner_SameMajority(tree)
        if getattr(self, "m_pruning", 0):
            tree = Pruner_m(tree, m=self.m_pruning)
        
        return octree.TreeClassifier(base_classifier=tree)


def convert_numpy_data_to_orange(orange_domain, X, y=None):
        """Convert the given X and y numpy arrays to an Orange data table
        with the given domain.
        If y is None (default), the class values of the Orange data table are
        set to '?'.
        
        Parameters
        ----------
        orange_domain : Orange.data.domain
            The domain of the newly created Orange data table. It should have
            the same number of features as the X numpy array.
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (integers that correspond to classes). 
        
        Returns
        -------
        orange_data : Orange.data.Table
            The data converted to an Orange data table.
        """
        if y == None:
            new_y = np.zeros((len(X), 1))
        else:
            new_y = np.atleast_2d(y).T
        X_y = np.hstack((X, new_y))
        orange_data = Orange.data.Table(orange_domain, X_y)
        # convert all class values from 0 to '?'
        if y == None:
            for ex in orange_data:
                ex.set_class('?')
        return orange_data


import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

class OrangeClassifierWrapper(BaseEstimator, ClassifierMixin):
    
    """A wrapper that wraps an Orange classification learner as a scikit-learn
    estimator.
    
    Parameters
    ----------
    orange_learner : a descendant of Orange.core.Leaner
        The Orange learner to be wrapped as a scikit-learn estimator.
    
    Attributes
    ----------
    `orange_classifier_` : a descendant of Orange.core.Classifier
        The underlying Orange classifier.
    
    `orange_data_` : Orange.data.Table 
        The Orange data table with the converted Numpy array learning data.
    
    """
    
    def __init__(self, orange_learner):
        if not isinstance(orange_learner, Orange.core.Learner):
            raise ValueError("The given orange_learner is not an Orange "
                             "learner.")
        self.orange_learner = orange_learner
        
        self.orange_classifier_ = None
    
    def fit(self, X, y):
        """Build a classifier for the training set (X, y).
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (integers that correspond to classes). 
        
        Returns
        -------
        self : object
            Returns self.
        """
        # convert numpy data to Orange
        self.n_features_ = X.shape[1]
        feat_names = ["a{}".format(k) for k in range(self.n_features_)]
        orange_feat = []
        for k in range(self.n_features_):
            if len(np.unique(X[:, k])) <= 2:
                feat = Orange.data.variable.Discrete(name=feat_names[k],
                                                     values=["0", "1"])
            else:
                feat = Orange.data.variable.Continuous(name=feat_names[k])
            orange_feat.append(feat)
        self.n_classes_ = len(np.unique(y))
        if self.n_classes_ != 2:
            raise ValueError("Only binary classification problems are "
                             "supported!")
        orange_class = Orange.data.variable.Discrete(name="cls",
                                                     values=["0", "1"])
        self.orange_domain_ = Orange.data.Domain(orange_feat, orange_class)
        self.orange_data_ = convert_numpy_data_to_orange(self.orange_domain_,
                                                         X, y)
        
        # build a classifier
        self.orange_classifier_ = self.orange_learner(self.orange_data_)
        
        return self

    def predict(self, X):
        """Predict class value for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        # perform sanity checks
        n_samples, n_features = X.shape
        if self.orange_classifier_ is None:
            raise Exception("OrangeClassifierWrapper not initialized. Call the"
                            " a fit() function first.")
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must match the "
                             "input. Model n_features is {} and input "
                             "n_features is {}".format(self.n_features_,
                                                       n_features))
        # convert numpy data to Orange
        orange_test_data = convert_numpy_data_to_orange(self.orange_domain_, X)
        # classify all examples with the previously built classifier
        y = np.empty(n_samples)
        for i, ex in enumerate(orange_test_data):
            y[i] = self.orange_classifier_(ex, Orange.core.GetValue)
        return y
        
    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by arithmetical order.
        """
        # perform sanity checks
        n_samples, n_features = X.shape
        if self.orange_classifier_ is None:
            raise Exception("OrangeClassifierWrapper not initialized. Call the"
                            " a fit() function first.")
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must match the "
                             "input. Model n_features is {} and input "
                             "n_features is {}".format(self.n_features_,
                                                       n_features))
        # convert numpy data to Orange
        orange_test_data = convert_numpy_data_to_orange(self.orange_domain_, X)
        # classify all examples with the previously built classifier
        p = np.empty((n_samples, self.n_classes_))
        for i, ex in enumerate(orange_test_data):
            p[i, :] = list(self.orange_classifier_(ex,
                                    Orange.core.GetProbabilities))
        return p


if __name__ == "__main__":
    data = Orange.data.Table("titanic")
    nt = octree.TreeLearner(data)
    print "'Normal' TreeLearner:"
    print nt # should have the 'sex' attribute as the first split
    print
    
    ffst = ForcedFirstSplitTreeLearner(data, first_split_attr="age")
    print "ForcedFirstSplitTreeLearner (with the first split forced to 'age'):"
    print ffst # should have 'age' attribute as the first split
    print

    import PyMTL.synthetic_data as sd
    a, d = 8, 4
    attr, func = sd.generate_boolean_function(a, d, random_seed=2)
    print "Boolean function (a={}, d={}): {}".format(a, d, func)
    X, y = sd.generate_examples(attr, func, n=100, random_state=10)
    print "% of True values in y: {:.2f}".format(100 * sum(y == True) / len(y))
    
    orange_learner = octree.TreeLearner()
    sklearn_wrapper = OrangeClassifierWrapper(orange_learner=orange_learner)
    
    from sklearn.cross_validation import cross_val_score
    print "Cross-validation CAs: ", cross_val_score(sklearn_wrapper, X, y=y)
    print "Cross-validation AUCs: ", cross_val_score(sklearn_wrapper, X, y=y,
                                                    scoring="roc_auc")
    