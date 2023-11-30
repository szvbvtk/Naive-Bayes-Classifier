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


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_prior = counts / y.size
        self.X = X
        self.y = y
        self.X_discretized = discretize_data(X, bins=self.bins)
        
        for class_ in self.classes:
            class_data = self.X_discretized[self.y == class_]
            self.feature_probs = dict()
            for feature_index in range(self.X_discretized.shape[1]):
                feature_values, feature_counts = np.unique(class_data[:, feature_index], return_counts=True)
                feature_probs = feature_counts / len(class_data)
                self.feature_probs[feature_index] = dict(zip(feature_values, feature_probs))

        return self
    
    def predict_proba(self, X):
        X_discretized = discretize_data(X, bins=self.bins)
        predictions = np.zeros((len(X), len(self.classes)))
        for sample_index in range(len(X)):
            sample = X_discretized[sample_index]
            for class_index, class_ in enumerate(self.classes):
                class_prior = self.class_prior[class_index]
                feature_probs = self.feature_probs
                likelihood = 1
                for feature_index in range(X_discretized.shape[1]):
                    feature_value = sample[feature_index]
                    if feature_value in feature_probs[feature_index]:
                        feature_prob = feature_probs[feature_index][feature_value]
                    else:
                        feature_prob = 0.0
                    likelihood *= feature_prob
                predictions[sample_index, class_index] = class_prior * likelihood
        return predictions
    
    def predict(self, X):
        predictions = self.predict_proba(X)
        return self.classes[np.argmax(predictions, axis=1)]
    

if __name__ == "__main__":
    data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

    bins = 15
    model = NaiveBayesClassifier(bins=bins)
    model.fit(X_train, y_train)
    print(model.predict(X_test))