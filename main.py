import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_prior = None
        self.classes = None
        self.feature_probs = dict()

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
                print(class_, feature_values, feature_counts)
                feature_probs = feature_counts / number_of_class_samples
                self.feature_probs[class_][feature_index] = dict(zip(feature_values, feature_probs))

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
                    print(class_, feature_index, feature_value)
                    feature_prob = feature_probs[feature_index][feature_value]
                    likelihood *= feature_prob

                predictions[sample_index, class_index] = class_prior * likelihood
            print(predictions)

        return predictions




    def predict(self, X):
        predictions = self.predict_proba(X)
        pass
        # return self.classes[np.argmax(predictions, axis=1)]






def discretize_data(data, bins, min= None, max = None):
    if min == None and max == None:
        min = np.min(data)
        max = np.max(data)

    bin_ranges = np.linspace(min, max, bins + 1)
    discretized_data = np.digitize(data, bin_ranges, right=True)

    return discretized_data

data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

bins = 30
X_train_discretized = discretize_data(X_train, bins)
X_test_discretized = discretize_data(X_test, bins)

model = NaiveBayesClassifier()
model.fit(X_train_discretized, y_train)
print(model.feature_probs[1])
# predictions = model.predict(X_test_discretized)

# a = np.arange(10)

# for i in a:
#     print(i)


# print('-------------------')
# for i in a:
#     print(i)

