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
        self.classes = np.unique(y)
        self.X = X
        self.y = y
        self.X_discretized = discretize_data(X, bins=self.bins)
        self.priors = np.zeros(len(self.classes))
        self.means = np.zeros((len(self.classes), self.X_discretized.shape[1]))
        self.stds = np.zeros((len(self.classes), self.X_discretized.shape[1]))
        for class_index, class_ in enumerate(self.classes):
            self.priors[class_index] = np.sum(self.y == class_) / len(self.y)
            self.means[class_index] = np.mean(self.X_discretized[self.y == class_], axis=0)
            self.stds[class_index] = np.std(self.X_discretized[self.y == class_], axis=0)
        return self
    
    def predict_proba(self, X):
        X_discretized = discretize_data(X, bins=self.bins)
        y_pred = np.zeros((len(X), len(self.classes)))
        for sample_index in range(len(X)):
            sample = X_discretized[sample_index]
            posteriors = np.zeros(len(self.classes))
            for class_index, class_ in enumerate(self.classes):
                prior = self.priors[class_index]
                mean = self.means[class_index]
                std = self.stds[class_index]
                likelihood = np.prod(np.exp(-((sample - mean) ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std))
                posteriors[class_index] = prior * likelihood
            y_pred[sample_index] = posteriors
        return y_pred
    
    def predict(self, X):
        y_pred = self.predict_proba(X)
        return self.classes[np.argmax(y_pred, axis=1)]

    # def predict(self, X):
    #     X_discretized = discretize_data(X, bins=self.bins)
    #     y_pred = np.zeros(len(X))
    #     for sample_index in range(len(X)):
    #         sample = X_discretized[sample_index]
    #         posteriors = np.zeros(len(self.classes))
    #         for class_index, class_ in enumerate(self.classes):
    #             prior = self.priors[class_index]
    #             mean = self.means[class_index]
    #             std = self.stds[class_index]
    #             likelihood = np.prod(np.exp(-((sample - mean) ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std))
    #             posteriors[class_index] = prior * likelihood
    #         y_pred[sample_index] = self.classes[np.argmax(posteriors)]
    #     return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)


data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

model = NaiveBayesClassifier()
model.fit(X_train, y_train)
print(model.predict(X_test))
# print(model.score(X_test, y_test))
