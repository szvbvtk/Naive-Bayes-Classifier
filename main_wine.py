import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier, score, discretize_numerical_data
from sklearn.model_selection import train_test_split




data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

X = data[:, 1:]
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

number_of_bins = 20
X_train_discretized = discretize_numerical_data(data=X_train, bins=number_of_bins, min_refs=np.min(X_test, axis=0), max_refs=np.max(X_test, axis=0))
X_test_discretized = discretize_numerical_data(data=X_test, bins=number_of_bins, min_refs=np.min(X_test, axis=0), max_refs=np.max(X_test, axis=0))

nbc = NaiveBayesClassifier(laplace_smoothing=False)
nbc.fit(X_train_discretized, y_train)

predictions = nbc.predict(X_test_discretized)
# print(predictions)
# print(y_test)
print(score(predictions, y_test))



