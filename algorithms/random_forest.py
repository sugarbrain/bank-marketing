# -*- coding: utf-8 -*-

import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
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

    param = {'n_estimators': 12, 
             'max_features': 6}
    
    clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                 max_features=param['max_features'],
                                 random_state=42)

    clf = clf.fit(X_train, y_train)

    print(f"Accuracy on split {split_num}: %0.3f" % clf.score(X_train, y_train))
    
    total_acc += clf.score(X_train, y_train)
    split_num += 1
        
print("\n=============================================")
print(f"KFold's mean: {total_acc/K_SPLITS}")
print("=============================================")