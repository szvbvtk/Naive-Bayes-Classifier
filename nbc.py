import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.stds = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = np.zeros(len(self.classes))
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.stds = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(X)
            self.means[i] = np.mean(X_c, axis=0)
            self.stds[i] = np.std(X_c, axis=0)

    def predict(self, X):
        posteriors = np.zeros((len(X), len(self.classes)))

        for i, x in enumerate(X):
            for j, c in enumerate(self.classes):
                prior = np.log(self.class_priors[j])
                likelihood = np.sum(-0.5 * np.log(2 * np.pi * self.stds[j]) - 0.5 * ((x - self.means[j]) / self.stds[j]) ** 2)
                posteriors[i, j] = prior + likelihood

        return self.classes[np.argmax(posteriors, axis=1)]
