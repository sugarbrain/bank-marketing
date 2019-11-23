# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection

# Tira limite de vizualição do dataframe quando printado
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

SEED = 42
np.random.seed(SEED)

# Full train set
train_file = "../datasets/train.csv"


def get_train_set(filepath, size=0.20):
    dataset = pandas.read_csv(train_file)

    test_size = 1.0 - size

    # use 20% of the train to search best params
    train, _ = train_test_split(dataset,
                                test_size=test_size,
                                random_state=SEED)

    return train


def get_estimators():
    # Classifiers with best params
    dt = tree.DecisionTreeClassifier(criterion='gini',
                                     max_depth=5,
                                     random_state=SEED)
    rf = RandomForestClassifier(n_estimators=29,
                                max_features='sqrt',
                                random_state=SEED)
    knn = KNeighborsClassifier(metric="euclidean", n_neighbors=13)
    mlpc = MLPClassifier(solver='sgd',
                         hidden_layer_sizes=(43, 2),
                         random_state=SEED)

    estimators = [
        ('DecisionTree', dt),
        ('RandomForestClassifier', rf),
        ('KNeighborsClassifier', knn),
        ('MLPClassifier', mlpc),
    ]

    return estimators


def fit_single_models(X, Y, estimators, kfold):
    results = list()

    for name, clf in estimators:
        result = model_selection.cross_val_score(clf, X, Y, cv=kfold)
        results.append((name, result.mean()))

    return results


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_committee_score(X, Y, estimators, kfold):

    # create the ensemble model
    ensemble = VotingClassifier(estimators, voting='hard')
    results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)

    return results.mean()


K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Get estimators
estimators = get_estimators()

single_model_results = fit_single_models(X, Y, estimators, kfold)

# Run scoring for best params
score = run_committee_score(X, Y, estimators, kfold)

for name, mean in single_model_results:
    print("%s result mean: %0.4f" % (name, mean))

print("Committee score: %0.4f" % score)
