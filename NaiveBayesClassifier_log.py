import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def score(y_predictions, y):
    return np.sum(y_predictions == y) / y.size



def discretize_numerical_data(data, bins, min_refs, max_refs):
    X_discretized = np.empty_like(data)
    for feature_index in range(data.shape[1]):
        feature = data[:, feature_index]

        feauture_bins = np.linspace(min_refs[feature_index], max_refs[feature_index], bins)
        X_discretized[:, feature_index] = np.digitize(feature, feauture_bins)

    return X_discretized

def discretize_categorical_data(data):
    encoded_data = data.copy()

    label_encoder = LabelEncoder()

    # Pętla po kolumnach
    for col_index in range(encoded_data.shape[1]):
        encoded_data[:, col_index] = label_encoder.fit_transform(encoded_data[:, col_index])

    return encoded_data

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
        priors = counts / y.size
        log_priors = np.log(priors)
        self.classes = dict(zip(labels, log_priors))
        number_of_features = X.shape[1]
        feature_indices = np.arange(number_of_features)
        for label in labels:
            labeldata = X[y == label]
            number_of_labelsamples = labeldata.shape[0]
            self.feature_log_probs[label] = dict()
            for feature_index in feature_indices:
                feature_values, feature_counts = np.unique(labeldata[:, feature_index], return_counts=True)

                if self.laplace_smoothing:
                    feature_probs = np.log((feature_counts + 1) / (number_of_labelsamples + feature_values.size))
                else:
                    feature_probs = np.log(feature_counts / number_of_labelsamples)
 
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
                log_likelihood = 0
                feature_log_probs = self.feature_log_probs[label]

                for feature_index in feature_indices:
                    feature_value = sample[feature_index]

                    log_feature_prob = feature_log_probs[feature_index][feature_value]
                    log_likelihood += log_feature_prob

                log_predictions[sample_index, label_index] = log_prior + log_likelihood

        return log_predictions

    def predict_proba(self, X):
        log_predictions = self.predict_log_proba(X)
        predictions = np.exp(log_predictions - np.max(log_predictions, axis=1).reshape(-1, 1)) / np.sum(np.exp(log_predictions - np.max(log_predictions, axis=1).reshape(-1, 1)), axis=1).reshape(-1, 1)

        return predictions

    def predict(self, X):
        log_predictions = self.predict_log_proba(X)
        labels = np.array(list(self.classes.keys()))
        return labels[np.argmax(log_predictions, axis=1)]