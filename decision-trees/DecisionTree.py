import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feature_indices = np.random.choice(num_features, self.num_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return TreeNode(best_feature, best_threshold, left, right)


    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_feature, split_threshold = None, None

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_index
                    split_threshold = threshold

        return split_feature, split_threshold


    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_indices, right_indices = self._split(X_column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
    
        num_samples = len(y)
        num_left, num_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (num_left / num_samples) * entropy_left + (num_right / num_samples) * entropy_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def _entropy(self, y):
        histogram = np.bincount(y)
        probabilities = histogram / len(y)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
