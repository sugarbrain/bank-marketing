# -*- coding: utf-8 -*-

import pandas
from sklearn import tree, preprocessing
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

split_num = 1

for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    params = [
        {
            "criterion": "gini",
            "max_depth": 7,
        },
        {
            "criterion": "gini",
            "max_depth": 5,
        },
        {
            "criterion": "gini",
            "max_depth": 8,
        },
        {
            "criterion": "gini",
            "max_depth": 12,
        },
        {
            "criterion": "gini",
            "max_depth": 3,
        },
        {
            "criterion": "entropy",
            "max_depth": 1,
        },
        {
            "criterion": "entropy",
            "max_depth": 5,
        },
        {
            "criterion": "entropy",
            "max_depth": 7,
        },
        {
            "criterion": "entropy",
            "max_depth": 12,
        },
        {
            "criterion": "entropy",
            "max_depth": 6,
        },
    ]
    
    for param in params:
        clf = tree.DecisionTreeClassifier(
                    criterion=param["criterion"],
                    max_depth=param["max_depth"],
                    random_state=42
                    )
    
        clf = clf.fit(X_train, y_train)

        print(param)    
        print(f"Accuracy on split {split_num}: %0.3f" % clf.score(X_train, y_train))
        split_num += 1