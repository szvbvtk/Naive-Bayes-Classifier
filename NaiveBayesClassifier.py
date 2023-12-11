import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

def score(y_predictions, y):
    return np.sum(y_predictions == y) / y.size



def discretize_data(data, bins, min_refs, max_refs):
    data_discretized = np.empty_like(data)
    for feature_index in np.arange(data.shape[1]):
        feature_data = data[:, feature_index]
        feauture_bins = np.linspace(min_refs[feature_index], max_refs[feature_index], bins)
        data_discretized[:, feature_index] = np.digitize(feature_data, feauture_bins)

    return data_discretized





class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, laplace_smoothing=False):
        self.laplace_smoothing = laplace_smoothing
        self.classes = None
        self.feature_probs = dict()

    @staticmethod
    def default_prob():
        return 1e-5

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts=True)
        priors = counts / y.size
        self.classes = dict(zip(labels, priors))
        number_of_features = X.shape[1]

        for label in labels:
            label_data = X[y == label]
            number_of_samples = label_data.shape[0]
            self.feature_probs[label] = dict()

            for feature_index in np.arange(number_of_features):
                feature_values, feature_counts = np.unique(label_data[:, feature_index], return_counts=True)
                print(feature_values)
                if self.laplace_smoothing:
                    feature_probs = (feature_counts + 1) / (number_of_samples + feature_values.size)
                else:
                    feature_probs = feature_counts / number_of_samples
 
                self.feature_probs[label][feature_index] = defaultdict(self.default_prob, zip(feature_values, feature_probs))

        return self
    

    
    def predict_proba(self, X):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        number_of_classes = len(self.classes.keys())

        predictions = np.empty((number_of_samples, number_of_classes))

        for sample_index, sample in enumerate(X):
            for label_index, label in enumerate(self.classes.keys()):
                prior = self.classes[label]
                feature_probs = self.feature_probs[label]

                for feature_index in np.arange(number_of_features):
                    feature_value = sample[feature_index]

                    feature_prob = feature_probs[feature_index][feature_value]
                    prior *= feature_prob

                predictions[sample_index, label_index] = prior
 

        return predictions

    def predict(self, X):
        predictions = self.predict_proba(X)
        labels = np.array(list(self.classes.keys()))
        return labels[np.argmax(predictions, axis=1)]