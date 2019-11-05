import numpy as np
from sklearn.model_selection import KFold
from numpy.random import seed

RANDOM_SEED = 42

seed(RANDOM_SEED)

def generate_indexes(dataset, k=10):
    X = dataset.values[:, 0:59]
    y = dataset.values[:,59]

    kf = KFold(n_splits=k)
    kf.get_n_splits(X)

    print(kf)

    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
