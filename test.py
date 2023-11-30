import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin




def discretize_data(X, bins=10):
    X_discretized = np.zeros_like(X)
    for feature_index in range(X.shape[1]):
        feature = X[:, feature_index]
        feature_min = feature.min()
        feature_max = feature.max()
        feature_bins = np.linspace(feature_min, feature_max, bins)
        X_discretized[:, feature_index] = np.digitize(feature, feature_bins)
    return X_discretized




data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)