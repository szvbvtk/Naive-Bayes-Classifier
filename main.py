import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, laplace_smoothing=False):
        self.laplace_smoothing = laplace_smoothing
        self.class_prior = None
        self.classes = None
        self.feature_probs = dict()

    @staticmethod
    def default_prob():
        return 1e-5

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        # print(counts)
        self.class_prior = counts / y.size
        number_of_features = X.shape[1]
        feature_indices = np.arange(number_of_features)
        for class_ in self.classes:
            class_data = X[y == class_]
            number_of_class_samples = class_data.shape[0]
            self.feature_probs[class_] = dict()
            for feature_index in feature_indices:
                feature_values, feature_counts = np.unique(class_data[:, feature_index], return_counts=True)
                # print(class_, feature_values, feature_counts)
                if self.laplace_smoothing:
                    feature_probs = (feature_counts + 1) / (number_of_class_samples + feature_values.size)
                else:
                    feature_probs = feature_counts / number_of_class_samples
                # self.feature_probs[class_][feature_index] = dict(zip(feature_values, feature_probs))
                self.feature_probs[class_][feature_index] = defaultdict(self.default_prob, zip(feature_values, feature_probs))

        # print(self.feature_probs)
        return self
    

    
    def predict_proba(self, X):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        number_of_classes = self.classes.size

        predictions = np.zeros((number_of_samples, number_of_classes))

        for sample_index, sample in enumerate(X):
            for class_index, class_ in enumerate(self.classes):
                class_prior = self.class_prior[class_index]
                feature_probs = self.feature_probs[class_]
                likelihood = 1

                for feature_index in range(number_of_features):
                    # print(class_, feature_index)
                    feature_value = sample[feature_index]
                    # print(class_, feature_index, feature_value)

                    # już niepotrzebne bo jest defaultdict
                    # if feature_value in feature_probs[feature_index]:
                    #     feature_prob = feature_probs[feature_index][feature_value]
                    # else:
                    #     feature_prob = 1e-5

                    feature_prob = feature_probs[feature_index][feature_value]
                    # feature_prob = feature_probs[feature_index][feature_value]
                    likelihood *= feature_prob

                predictions[sample_index, class_index] = class_prior * likelihood
            # print(predictions)

        return predictions

    def predict(self, X):
        predictions = self.predict_proba(X)
        return self.classes[np.argmax(predictions, axis=1)]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)

    
    # def predict_proba(self, X):
    #     predictions = []
    #     self.class_priors = dict(zip(model.classes, model.class_prior))
    #     for sample in X:
    #         class_probs = {}
    #         for c in self.class_priors:
    #             prob = self.class_priors[c]
    #             for feature_index, feature_value in enumerate(sample):
    #                 if feature_value in self.feature_probs[c][feature_index]:
    #                     prob *= self.feature_probs[c][feature_index][feature_value]
    #             class_probs[c] = prob
    #         predictions.append(class_probs)
    #     return predictions




    # def predict(self, X):
    #     predictions = self.predict_proba(X)
    #     return [max(p, key=p.get) for p in predictions]
    #     return self.classes[np.argmax(predictions, axis=1)]






def discretize_data(data, bins, min_val=None, max_val=None):
    X_discretized = np.zeros_like(data)

    for feature_index in range(data.shape[1]):
        feature = data[:, feature_index]
        feature_min = feature.min()
        feature_max = feature.max()

        feauture_bins = np.linspace(feature_min, feature_max, bins)
        X_discretized[:, feature_index] = np.digitize(feature, feauture_bins)

    return X_discretized

def discretize_data2(X, bins=10):
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

bins = 30
X_train_discretized = discretize_data(X_train, bins)
X_test_discretized = discretize_data(X_test, bins)

model = NaiveBayesClassifier(laplace_smoothing=True)
model.fit(X_train_discretized, y_train)
# print(model.feature_probs[1][12].keys())
predictions = model.predict(X_test_discretized)
print(predictions)
print(y_test)
print(model.score(X_test_discretized, y_test))
# print(dict(zip(model.classes, model.class_prior)))


