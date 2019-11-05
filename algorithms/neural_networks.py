# -*- coding: utf-8 -*-

import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

K_SPLITS = 10

file_name = "../bank_additional_full.csv"

dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")

dataset.fillna(method='ffill', inplace=True)

ss = preprocessing.StandardScaler()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset = pandas.get_dummies(dataset, columns=[column_name])
    else:
        dataset[column_name] = ss.fit_transform(dataset[[column_name]])

X = dataset.values[:, 0:-2]
Y = dataset['y_no']

kf = KFold(n_splits=K_SPLITS)
kf.get_n_splits(X)

total_acc = 0
split_num = 1

for train_index, test_index in kf.split(X):
   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    param = {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    }
    
    clf = MLPClassifier(solver=param["solver"],
                        hidden_layer_sizes=param["hidden_layer_sizes"],
                        activation=param["activation"],
                        random_state=42)

    clf = clf.fit(X_train, y_train)

    print(f"Accuracy on split {split_num}: %0.3f" % clf.score(X_train, y_train))
    
    total_acc += clf.score(X_train, y_train)
    split_num += 1
        
print("\n=============================================")
print(f"KFold's mean: {total_acc/K_SPLITS}")
print("=============================================")
