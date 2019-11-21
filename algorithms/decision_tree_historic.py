# -*- coding: utf-8 -*-

import pandas
from sklearn import tree, preprocessing
from sklearn.model_selection import KFold

SEED = 42

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

params = []
    
for depth in range(1, 12):
    params.append({
                "criterion": "gini",
                "max_depth": depth
            })
for depth in range(1, 12):
    params.append({
                "criterion": "entropy",
                "max_depth": depth
            })

best_acc_mean = 0
best_param = {}

for param in params:
    print(f"\nUsing param -> {param}\n")
    
    total_acc = 0
    split_num = 1
    
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = tree.DecisionTreeClassifier(
                criterion=param["criterion"],
                max_depth=param["max_depth"],
                random_state=SEED
                )

        clf = clf.fit(X_train, y_train)
        
        score = clf.score(X_train, y_train)

        print(f"Accuracy on split {split_num}: %0.3f" % score)
        
        total_acc += score
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