# -*- coding: utf-8 -*-

import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

K_SPLITS = 10

file_name = '../bank_additional_full.csv'

dataset = pandas.read_csv(file_name, sep=';', na_values='unknown')

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


params = [
        {
            'n_estimators': 10,
            'max_features': 7
        },
        {
            'n_estimators': 16,
            'max_features': None
        },
        {
            'n_estimators': 4,
            'max_features': 8
        },
        {
            'n_estimators': 8,
            'max_features': 10
        },
        {
            'n_estimators': 4,
            'max_features': 7
        },
        {
            'n_estimators': 12,
            'max_features': 6,
        },
        {
            'n_estimators': 6,
            'max_features': 12,
        },
        {
            'n_estimators': 5,
            'max_features': 7,
        },
        {
            'n_estimators': 7,
            'max_features': 4,
        },
        {
            'n_estimators': 9,
            'max_features': 6,
        },
    ]

best_acc_mean = 0
best_param = {}

for param in params:
    print(f"\nUsing param -> {param}\n")
    
    total_acc = 0
    split_num = 1

    for train_index, test_index in kf.split(X):
   
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                     max_features=param['max_features'],
                                     random_state=42)
    
        clf = clf.fit(X_train, y_train)

        print(param)

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
