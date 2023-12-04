import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

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
        feature_indices = np.arange(number_of_features)
        for label in labels:
            labeldata = X[y == label]
            number_of_labelsamples = labeldata.shape[0]
            self.feature_probs[label] = dict()
            for feature_index in feature_indices:
                feature_values, feature_counts = np.unique(labeldata[:, feature_index], return_counts=True)

                if self.laplace_smoothing:
                    feature_probs = (feature_counts + 1) / (number_of_labelsamples + feature_values.size)
                else:
                    feature_probs = feature_counts / number_of_labelsamples
 
                self.feature_probs[label][feature_index] = defaultdict(self.default_prob, zip(feature_values, feature_probs))

        return self
    

    
    def predict_proba(self, X):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        feature_indices = np.arange(number_of_features)
        number_of_classes = len(self.classes.keys())

        predictions = np.empty((number_of_samples, number_of_classes))

        for sample_index, sample in enumerate(X):
            for label_index, label in enumerate(self.classes.keys()):
                prior = self.classes[label]
                feature_probs = self.feature_probs[label]
                likelihood = 1

                for feature_index in feature_indices:
                    feature_value = sample[feature_index]

                    # już nie jest potrzebne bo jest defaultdict
                    # if feature_value in feature_probs[feature_index]:
                    #     feature_prob = feature_probs[feature_index][feature_value]
                    # else:
                    #     feature_prob = 1e-5

                    feature_prob = feature_probs[feature_index][feature_value]
                    prior *= feature_prob

                predictions[sample_index, label_index] = prior
 

        return predictions

    def predict(self, X):
        predictions = self.predict_proba(X)
        labels = np.array(list(self.classes.keys()))
        return labels[np.argmax(predictions, axis=1)]



def score(y_predictions, y):
    return np.sum(y_predictions == y) / y.size



def discretize_data(data, bins, min_refs, max_refs):
    X_discretized = np.empty_like(data)
    for feature_index in range(data.shape[1]):
        feature = data[:, feature_index]

        feauture_bins = np.linspace(min_refs[feature_index], max_refs[feature_index], bins)
        X_discretized[:, feature_index] = np.digitize(feature, feauture_bins)

    return X_discretized

data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

number_of_bins = 20
X_train_discretized = discretize_data(data=X_train, bins=number_of_bins, min_refs=np.min(X_test, axis=0), max_refs=np.max(X_test, axis=0))
X_test_discretized = discretize_data(data=X_test, bins=number_of_bins, min_refs=np.min(X_test, axis=0), max_refs=np.max(X_test, axis=0))

model = NaiveBayesClassifier(laplace_smoothing=True)
model.fit(X_train_discretized, y_train)

predictions = model.predict(X_test_discretized)
print(predictions)
print(y_test)
print(score(predictions, y_test))



