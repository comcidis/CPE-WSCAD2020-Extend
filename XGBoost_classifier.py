from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import progressbar
from mlfromscratch.utils import train_test_split, standardize, to_categorical, normalize,divide_on_feature
from mlfromscratch.utils import mean_squared_error, Plot, calculate_entropy, accuracy_score, calculate_variance
import pyRAPL
from mlfromscratch.deep_learning.activation_functions import Sigmoid
from mlfromscratch.utils.misc import bar_widgets
import pandas as pd
pyRAPL.setup()
csv_output=pyRAPL.outputs.CSVOutput('Hotspots_XGBoost_classifier.csv')

class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        with pyRAPL.Measurement('Fit_1',output=csv_output):
            """ Build decision tree """
            self.one_dim = len(np.shape(y)) == 1
            self.root = self._build_tree(X, y)
            self.loss=None
        csv_output.save()

    def _build_tree(self, X, y, current_depth=0):
        with pyRAPL.Measurement('Build_tree',output=csv_output):
            """ Recursive method which builds out the decision tree and splits X and respective y
            on the feature of X which (based on impurity) best separates the data"""

            largest_impurity = 0
            best_criteria = None    # Feature index and threshold
            best_sets = None        # Subsets of the data

            # Check if expansion of y is needed
            if len(np.shape(y)) == 1:
                y = np.expand_dims(y, axis=1)

            # Add y as last column of X
            Xy = np.concatenate((X, y), axis=1)

            n_samples, n_features = np.shape(X)

            if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
                # Calculate the impurity for each feature
                for feature_i in range(n_features):
                    # All values of feature_i
                    feature_values = np.expand_dims(X[:, feature_i], axis=1)
                    unique_values = np.unique(feature_values)

                    # Iterate through all unique values of feature column i and
                    # calculate the impurity
                    for threshold in unique_values:
                        # Divide X and y depending on if the feature value of X at index feature_i
                        # meets the threshold
                        Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                        if len(Xy1) > 0 and len(Xy2) > 0:
                            # Select the y-values of the two sets
                            y1 = Xy1[:, n_features:]
                            y2 = Xy2[:, n_features:]

                            # Calculate impurity
                            impurity = self._impurity_calculation(y, y1, y2)

                            # If this threshold resulted in a higher information gain than previously
                            # recorded save the threshold value and the feature
                            # index
                            if impurity > largest_impurity:
                                largest_impurity = impurity
                                best_criteria = {"feature_i": feature_i, "threshold": threshold}
                                best_sets = {
                                    "leftX": Xy1[:, :n_features],   # X of left subtree
                                    "lefty": Xy1[:, n_features:],   # y of left subtree
                                    "rightX": Xy2[:, :n_features],  # X of right subtree
                                    "righty": Xy2[:, n_features:]   # y of right subtree
                                    }

            if largest_impurity > self.min_impurity:
                # Build subtrees for the right and left branches
                true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
                false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
                return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                    "threshold"], true_branch=true_branch, false_branch=false_branch)

            # We're at leaf => determine value
            leaf_value = self._leaf_value_calculation(y)

            return DecisionNode(value=leaf_value)
        csv_output.save()

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """
        with pyRAPL.Measurement('Predict_value',output=csv_output):
            if tree is None:
                tree = self.root

            # If we have a value (i.e we're at a leaf) => return value as the prediction
            if tree.value is not None:
                return tree.value

            # Choose the feature that we will test
            feature_value = x[tree.feature_i]

            # Determine if we will follow left or right branch
            branch = tree.false_branch
            if isinstance(feature_value, int) or isinstance(feature_value, float):
                if feature_value >= tree.threshold:
                    branch = tree.true_branch
            elif feature_value == tree.threshold:
                branch = tree.true_branch

            # Test subtree
            return self.predict_value(x, branch)
        csv_output.save()

    def predict(self, X):
        with pyRAPL.Measurement('Predict_1',output=csv_output):
            """ Classify samples one by one and return the set of labels """
            y_pred = [self.predict_value(sample) for sample in X]
            return y_pred
        csv_output.save()

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)

class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost
    - Reference -
    http://xgboost.readthedocs.io/en/latest/model.html
    """

    def _split(self, y):
        with pyRAPL.Measurement('Split',output=csv_output):
           """ y contains y_true in left half of the middle column and
           y_pred in the right half. Split and return the two matrices """
           col = int(np.shape(y)[1]/2)
           y, y_pred = y[:, :col], y[:, col:]
           return y, y_pred
        csv_output.save()

    def _gain(self, y, y_pred):
        with pyRAPL.Measurement('Gain',output=csv_output):
           nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
           denominator = self.loss.hess(y, y_pred).sum()
           return 0.5 * (nominator / denominator)
        csv_output.save()

    def _gain_by_taylor(self, y, y1, y2):
        with pyRAPL.Measurement('Gain_by_taylor',output=csv_output):
           # Split
           y, y_pred = self._split(y)
           y1, y1_pred = self._split(y1)
           y2, y2_pred = self._split(y2)

           true_gain = self._gain(y1, y1_pred)
           false_gain = self._gain(y2, y2_pred)
           gain = self._gain(y, y_pred)
           return true_gain + false_gain - gain
        csv_output.save()

    def _approximate_update(self, y):
        with pyRAPL.Measurement('Approximate_update',output=csv_output):
           # y split into y, y_pred
           y, y_pred = self._split(y)
           # Newton's Method
           gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
           hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
           update_approximation =  gradient / hessian

           return update_approximation
        csv_output.save()

    def fit(self, X, y):
        with pyRAPL.Measurement('Fit_2',output=csv_output):
           self._impurity_calculation = self._gain_by_taylor
           self._leaf_value_calculation = self._approximate_update
           super(XGBoostRegressionTree, self).fit(X, y)
        csv_output.save()

class LogisticLoss():
    def __init__(self):
       sigmoid = Sigmoid()
       self.log_func = sigmoid
       self.log_grad = sigmoid.gradient

    def loss(self, y, y_pred):
       with pyRAPL.Measurement('Loss',output=csv_output):
          y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
          p = self.log_func(y_pred)
          return y * np.log(p) + (1 - y) * np.log(1 - p)
       csv_output.save()

    # gradient w.r.t y_pred
    def gradient(self, y, y_pred):
       with pyRAPL.Measurement('Gradient',output=csv_output):
          p = self.log_func(y_pred)
          return -(y - p)
       csv_output.save()

    # w.r.t y_pred
    def hess(self, y, y_pred):
       with pyRAPL.Measurement('Hess',output=csv_output):
          p = self.log_func(y_pred)
          return p * (1 - p)
       csv_output.save()

class XGBoost(object):
    """The XGBoost classifier.

    Reference: http://xgboost.readthedocs.io/en/latest/model.html

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further. 
    max_depth: int
        The maximum depth of a tree.
    """
    def __init__(self, n_estimators=200, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators            # Number of trees
        self.learning_rate = learning_rate          # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity              # Minimum variance reduction to continue
        self.max_depth = max_depth                  # Maximum depth for tree

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # Log loss for classification
        self.loss = LogisticLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth,
                    loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y):
        with pyRAPL.Measurement('Fit_3',output=csv_output):
           y = to_categorical(y)

           y_pred = np.zeros(np.shape(y))
           for i in self.bar(range(self.n_estimators)):
               tree = self.trees[i]
               y_and_pred = np.concatenate((y, y_pred), axis=1)
               tree.fit(X, y_and_pred)
               update_pred = tree.predict(X)

               y_pred -= np.multiply(self.learning_rate, update_pred)
        csv_output.save()

    def predict(self, X):
        with pyRAPL.Measurement('Predict_2',output=csv_output):
           y_pred = None
           # Make predictions
           for tree in self.trees:
               # Estimate gradient and update prediction
               update_pred = tree.predict(X)
               if y_pred is None:
                   y_pred = np.zeros_like(update_pred)
               y_pred -= np.multiply(self.learning_rate, update_pred)

           # Turn into probability distribution (Softmax)
           y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
           # Set label to the value that maximizes probability
           y_pred = np.argmax(y_pred, axis=1)
           return y_pred
        csv_output.save()

def main():

    print ("-- XGBoost --")

    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder

    with pyRAPL.Measurement('Read_data',output=csv_output):
        df = pd.read_csv('/home/gabi/Teste/BaseSintetica/1k_5att.csv')

        X = df.iloc[:, :-1].values
        y = df.iloc[:,-1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)

    csv_output.save()

    clf = XGBoost()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
