import numpy as np
from NaiveBayesClassifier import NaiveBayesClassifier, score
from NaiveBayesClassifier_log import NaiveBayesClassifier_log
from sklearn.model_selection import train_test_split

data = np.genfromtxt('Datasets/Mushroom/agaricus-lepiota.data', delimiter=',', dtype='U1') # na koniec zmienić na S1 i spr czy działa

X = data[:, 1:]
# X_discretized = discretize_categorical_data(X)
y = data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

nbc = NaiveBayesClassifier_log(laplace_smoothing=True)
nbc.fit(X_train, y_train)

predictions = nbc.predict(X_test)
print(score(predictions, y_test))


