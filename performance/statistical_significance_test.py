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
from scipy.stats import kruskal

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

    train, _ = train_test_split(dataset,
                                test_size=test_size,
                                random_state=SEED)

    return dataset


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


def get_kfold_scores_from_estimators(X, Y, estimators, kfold):
    results = list()

    for name, clf in estimators:
        result = model_selection.cross_val_score(clf, X, Y, cv=kfold)
        results.append((name, result))

    return results


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_significance_test(scores):

    print(scores)

    stat, p = kruskal(*[score[1] for score in scores])
    print('Kruskal-Wallis Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')

    # for name, score in scores:
    #     for name_inner, score_inner in scores:
    #         if name != name_inner:
    #             stat, p = kruskal(score, score_inner)
    #             alpha = 0.05
    #             print(f"\nComparing {name} with {name_inner}")
    #             print('Kruskal-Wallis Statistics=%.3f, p=%.3f' % (stat, p))
    #             if p > alpha:
    #                 print('Same distributions (fail to reject H0)')
    #             else:
    #                 print('Different distributions (reject H0)')


K_SPLITS = 10

# split train set by 80%
train = get_train_set(train_file, 0.8)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Get estimators
estimators = get_estimators()

scores = get_kfold_scores_from_estimators(X, Y, estimators, kfold)

run_significance_test(scores)
