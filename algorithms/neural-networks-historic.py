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

params = [
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "identity",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "logistic",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 3),
        "activation": "relu",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 3),
        "activation": "relu",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (3, 30),
        "activation": "relu",
    },
    {
        "solver": "lbfgs",
        "hidden_layer_sizes": (30, 30),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "identity",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "logistic",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 3),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 3),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (3, 30),
        "activation": "relu",
    },
    {
        "solver": "sgd",
        "hidden_layer_sizes": (30, 30),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "identity",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "logistic",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "tanh",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 3),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 3),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (3, 30),
        "activation": "relu",
    },
    {
        "solver": "adam",
        "hidden_layer_sizes": (30, 30),
        "activation": "relu",
    },
]


kf = KFold(n_splits=K_SPLITS)
kf.get_n_splits(X)

best_acc_mean = 0
best_param = {}

for param in params:
    
    print(f"\nUsing param -> {param}\n")
    
    total_acc = 0
    split_num = 1
    
    for train_index, test_index in kf.split(X):
   
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = MLPClassifier(solver=param["solver"],
                            hidden_layer_sizes=param["hidden_layer_sizes"],
                            activation=param["activation"],
                            random_state=42)
    
        clf = clf.fit(X_train, y_train)
    
        print(f"Accuracy on split {split_num}: %0.3f" % clf.score(X_train, 
                                                                  y_train))
        
        total_acc += clf.score(X_train, y_train)
        split_num += 1
    
    acc_mean = total_acc / K_SPLITS
            
    print("\n=============================================")
    print(f"KFold's mean: {acc_mean}")
    print("=============================================\n")
    
    if acc_mean > best_acc_mean:
        best_acc_mean = acc_mean
        best_param = param
        
print(f"Best accuracy mean: {best_acc_mean}")
print(f"Best param: {best_param}")
    