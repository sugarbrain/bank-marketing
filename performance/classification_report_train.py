# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Tira limite de vizualição do dataframe quando printado
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

SEED = 42
np.random.seed(SEED)

# Full train set
train_file = "../datasets/train.csv"


def get_dataset(filepath):
    dataset = pandas.read_csv(train_file)

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

    heterogen_committee_estimators = [
        ('DecisionTree', dt),
        ('RandomForestClassifier', rf),
        ('KNeighborsClassifier', knn),
        ('MLPClassifier', mlpc),
    ]

    # Heterogen Committee
    hc = VotingClassifier(heterogen_committee_estimators, voting='soft')

    # Neural Networks Committee
    sgd43_2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(43, 2))  # 0.908
    sgd6_1 = MLPClassifier(solver='sgd', hidden_layer_sizes=(6, 1))  # 0.907
    sgd18_2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(18, 2))  # 0.907
    sgd36_1 = MLPClassifier(solver='sgd', hidden_layer_sizes=(36, 1))  # 0.907
    ada2_1 = MLPClassifier(solver='adam', hidden_layer_sizes=(2, 1))  # 0.907

    neural_networks_estimators = [
        ('sgd43_2', sgd43_2),
        ('sgd6_1', sgd6_1),
        ('sgd18_2', sgd18_2),
        ('sgd36_1', sgd36_1),
        ('ada2_1', ada2_1),
    ]

    nnc = VotingClassifier(neural_networks_estimators, voting='hard')

    estimators = [
        ('DecisionTree', dt),
        ('RandomForestClassifier', rf),
        ('KNeighborsClassifier', knn),
        ('MLPClassifier', mlpc),
        ('HeterogenCommitteeVotingClassifier', hc),
        ('NeuralNetworksCommitteeVotingClassifier', nnc),
    ]

    return estimators


def run_classification_report(estimators, X_train, X_test, y_train, y_test):
    for name, clf in estimators:
        clf = clf.fit(X_train, y_train)
        Y_test_prediction = clf.predict(X_test)
        print(f"""
=> {name} classifier report <=

* Classification report:
{classification_report(y_test, Y_test_prediction)}

* Confusion matrix:
{confusion_matrix(y_test, Y_test_prediction)}

* Accuracy:
  - Train split accuracy: {clf.score(X_train, y_train)}
  - Test split accuracy: {clf.score(X_test, y_test)}
""")


K_SPLITS = 10

dataset = get_dataset(train_file)

# separate class from other columns
X = dataset.values[:, :-1]
Y = dataset['y']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                    random_state=SEED)

# Get estimators
estimators = get_estimators()

run_classification_report(estimators, X_train, X_test, y_train, y_test)
