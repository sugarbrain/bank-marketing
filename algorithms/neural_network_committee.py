# Voting Ensemble for Classification
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

K_SPLITS = 10

file_name = "../datasets/bank_additional_full.csv"

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

seed = 42
kfold = model_selection.StratifiedKFold(n_splits=K_SPLITS, random_state=seed)

# create single models
mlp44 = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30), activation='tanh', random_state=seed) # 0.946
mlp48 = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30), activation='relu', random_state=seed) # 0.943
mlp42 = MLPClassifier(solver='adam', hidden_layer_sizes=(30,3), activation='tanh', random_state=seed)  # 0.937
mlp16 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,3), activation='relu', random_state=seed) # 0.936
mlp10 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,3), activation='tanh', random_state=seed) # 0.933

#Fitting single models
results = model_selection.cross_val_score(mlp44, X, Y, cv=kfold)
print('MLP44: ', results.mean())

results = model_selection.cross_val_score(mlp48, X, Y, cv=kfold)
print('MLP48: ', results.mean())

results = model_selection.cross_val_score(mlp42, X, Y, cv=kfold)
print('MLP42: ', results.mean())

results = model_selection.cross_val_score(mlp16, X, Y, cv=kfold)
print('MLP16: ', results.mean())

results = model_selection.cross_val_score(mlp10, X, Y, cv=kfold)
print('MLP10: ', results.mean())

# create the sub models
estimators = [
    ('mlp44', mlp44),
    ('mlp48', mlp48),
    ('mlp42', mlp42),
    ('mlp16', mlp16),
    ('mlp10', mlp10),
]

# create the ensemble model
ensemble = VotingClassifier(estimators, voting='hard')
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
