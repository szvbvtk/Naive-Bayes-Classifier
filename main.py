from numpy import genfromtxt
from NBC import *
from sklearn.model_selection import train_test_split

def main_wine(laplace=False, log=False, continuous=False, _number_of_bins=10):
    data = genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)
    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    if continuous:
        X = X[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

        nbc = NaiveBayesClassifier_continuous()
        predictions = nbc.fit(X_train, y_train).predict(X_test)
    else:

        X_train_discretized, X_test_discretized = discretize(X, X_train, X_test, number_of_bins=_number_of_bins)

        
        if log:
            nbc = NaiveBayesClassifier_discrete_log(laplace_correction=laplace)
        else:
            nbc = NaiveBayesClassifier_discrete(laplace_correction=laplace)

        predictions = nbc.fit(X_train_discretized, y_train).predict(X_test_discretized)

    print(f"wine - {nbc.__class__.__name__}: {round(score(predictions, y_test), 4)}")

def main_mushroom(laplace=False, log=False):
    data = genfromtxt('Datasets/Mushroom/agaricus-lepiota.data', delimiter=',', dtype='S1')
    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=212121)

    if log:
        nbc = NaiveBayesClassifier_discrete_log(laplace_correction=laplace)
    else:
        nbc = NaiveBayesClassifier_discrete(laplace_correction=laplace)

    nbc.fit(X_train, y_train)

    predictions = nbc.predict(X_test)
    print(f"mushroom - {nbc.__class__.__name__}: {round(score(predictions, y_test), 4)}")


if __name__ == '__main__':
    # main_wine(laplace=False, log=False, continuous=False, _number_of_bins=10)
    main_mushroom(laplace=True, log=True)

