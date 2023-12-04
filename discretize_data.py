from sklearn.model_selection import train_test_split

import numpy as np
def discretize_data(data, bins):
    X_discretized = np.empty_like(data)

    for feature_index in range(data.shape[1]):
        print(feature_index)
        feature = data[:, feature_index]
        feature_min = feature.min()
        feature_max = feature.max()

        feauture_bins = np.linspace(feature_min, feature_max, bins)
        X_discretized[:, feature_index] = np.digitize(feature, feauture_bins)

    return X_discretized


def discretize_data2(data, bins, min_refs, max_refs):
    X_discretized = np.zeros_like(data)
    for feature_index in range(data.shape[1]):
        feature = data[:, feature_index]


        feauture_bins = np.linspace(min_refs[feature_index], max_refs[feature_index], bins)
        X_discretized[:, feature_index] = np.digitize(feature, feauture_bins)

    return X_discretized


# data = np.genfromtxt('Datasets/Wine/wine.data', delimiter=',', dtype=np.float16)

# X = data[:, 1:]
# y = data[:, 0]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# print(np.min(X_test, axis=0)[0])