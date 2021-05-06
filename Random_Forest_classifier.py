import pyRAPL
import pandas as pd
import numpy as np
from sklearn import datasets
from mlfromscratch.utils import train_test_split
from mlfromscratch.utils import get_random_subsets, normalize, standardize, calculate_entropy,accuracy_score
from mlfromscratch.utils import mean_squared_error, calculate_variance, divide_on_feature, Plot
import math
import progressbar
from mlfromscratch.utils.misc import bar_widgets

pyRAPL.setup()
csv_output=pyRAPL.outputs.CSVOutput('Hotspots_Random_forest_classifier.csv')

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
            """ Recursive method which builds out the decision tree and splits X and respective y on the feature of X which (based on impurity) best separates the data"""
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
        with pyRAPL.Measurement('Predict_value',output=csv_output):
            """ Do a recursive search down the tree and make a prediction of the data sample by the
                value of the leaf that we end up at """

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
        with pyRAPL.Measurement('Print_tree',output=csv_output):
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
        csv_output.save()



class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        with pyRAPL.Measurement('Information_gain',output=csv_output):
           # Calculate information gain
           p = len(y1)/len(y)
           entropy = calculate_entropy(y)
           info_gain = entropy - p * \
              calculate_entropy(y1) - (1 - p) * \
              calculate_entropy(y2)
           return info_gain
        csv_output.save()

    def _majority_vote(self, y):
        with pyRAPL.Measurement('Majority_vote',output=csv_output):
           most_common = None
           max_count = 0
           for label in np.unique(y):
              # Count number of occurences of samples with label
              count = len(y[y == label])
              if count > max_count:
                 most_common = label
                 max_count = count
           return most_common
        csv_output.save()

    def fit(self, X, y):
        with pyRAPL.Measurement('Fit_2',output=csv_output):
           self._impurity_calculation = self._calculate_information_gain
           self._leaf_value_calculation = self._majority_vote
           super(ClassificationTree, self).fit(X, y)
        csv_output.save()

class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further. 
    max_depth: int
        The maximum depth of a tree.
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators    # Number of trees
        self.max_features = max_features    # Maxmimum number of features per tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain            # Minimum information gain req. to continue
        self.max_depth = max_depth          # Maximum depth for tree
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))


    def fit(self, X, y):
        with pyRAPL.Measurement('Fit_3',output=csv_output):
           n_features = np.shape(X)[1]
           # If max_features have not been defined => select it as
           # sqrt(n_features)
           if not self.max_features:
              self.max_features = int(math.sqrt(n_features))

           # Choose one random subset of the data for each tree
           subsets = get_random_subsets(X, y, self.n_estimators)

           for i in self.progressbar(range(self.n_estimators)):
              X_subset, y_subset = subsets[i]
              # Feature bagging (select random subsets of the features)
              idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
              # Save the indices of the features for prediction
              self.trees[i].feature_indices = idx
              # Choose the features corresponding to the indices
              X_subset = X_subset[:, idx]
              # Fit the tree to the data
              self.trees[i].fit(X_subset, y_subset)
        csv_output.save()

    def predict(self, X):
        with pyRAPL.Measurement('Predict_2',output=csv_output):
           y_preds = np.empty((X.shape[0], len(self.trees)))
           # Let each tree make a prediction on the data
           for i, tree in enumerate(self.trees):
              # Indices of the features that the tree has trained on
              idx = tree.feature_indices
              # Make a prediction based on those features
              prediction = tree.predict(X[:, idx])
              y_preds[:, i] = prediction
           y_pred = []
           # For each sample
           for sample_predictions in y_preds:
              # Select the most common class prediction
              y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
           return y_pred
        csv_output.save()



def main():
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder

    with pyRAPL.Measurement('Read_data',output=csv_output):
        df = pd.read_csv('/home/gabi/Teste/BaseSintetica/1k_5att.csv')

        X = df.iloc[:,0:5].values
        y = df.iloc[:,5].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)
    csv_output.save()

    clf = RandomForest(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()



