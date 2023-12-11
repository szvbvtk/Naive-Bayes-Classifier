import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

def score(y_predictions, y):
    return np.sum(y_predictions == y) / y.size

class NaiveBayesClassifier_log(BaseEstimator, ClassifierMixin):
    def __init__(self, laplace_smoothing=False):
        self.laplace_smoothing = laplace_smoothing
        self.classes = None
        self.feature_log_probs = dict()

    @staticmethod
    def default_log_prob():
        return np.log(1e-5)

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts=True)
        log_priors = np.log(counts / y.size)
        self.classes = dict(zip(labels, log_priors))
        number_of_features = X.shape[1]

        for label in labels:
            label_data = X[y == label]
            number_of_samples = label_data.shape[0]
            self.feature_log_probs[label] = dict()
            for feature_index in np.arange(number_of_features):
                feature_values, feature_counts = np.unique(label_data[:, feature_index], return_counts=True)

                if self.laplace_smoothing:
                    feature_probs = np.log((feature_counts + 1) / (number_of_samples + feature_values.size))
                else:
                    feature_probs = np.log(feature_counts / number_of_samples)
 
                self.feature_log_probs[label][feature_index] = defaultdict(self.default_log_prob, zip(feature_values, feature_probs))

        return self

    def predict_log_proba(self, X):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        feature_indices = np.arange(number_of_features)
        number_of_classes = len(self.classes.keys())

        log_predictions = np.empty((number_of_samples, number_of_classes))

        for sample_index, sample in enumerate(X):
            for label_index, label in enumerate(self.classes.keys()):
                log_prior = self.classes[label]
                feature_log_probs = self.feature_log_probs[label]

                for feature_index in feature_indices:
                    feature_value = sample[feature_index]

                    log_feature_prob = feature_log_probs[feature_index][feature_value]
                    log_prior += log_feature_prob

                log_predictions[sample_index, label_index] = log_prior

        return log_predictions

    def predict(self, X):
        log_predictions = self.predict_log_proba(X)
        labels = np.array(list(self.classes.keys()))
        return labels[np.argmax(log_predictions, axis=1)]