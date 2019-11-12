# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier

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

seed = 42
kfold = model_selection.KFold(n_splits=K_SPLITS, random_state=seed)

# create single models
rf = RandomForestClassifier(n_estimators=16, max_features=None, random_state=seed)
knn = KNeighborsClassifier(metric="euclidean", n_neighbors=11)
mlpc = MLPClassifier(solver='adam', hidden_layer_sizes=(30, 30), 
                     activation="tanh", random_state=seed)
rg = RidgeClassifier(random_state=seed)

#Fitting single models
#results = model_selection.cross_val_score(rf, X, Y, cv=kfold)
#print('RandomForestClassifier: ', results.mean())

#results = model_selection.cross_val_score(knn, X, Y, cv=kfold)
#print('KNeighborsClassifier: ', results.mean())

#results = model_selection.cross_val_score(mlpc, X, Y, cv=kfold)
#print('MLPClassifier: ', results.mean())

#results = model_selection.cross_val_score(rg, X, Y, cv=kfold)
#print('RidgeClassifier: ', results.mean())

# create the sub models
estimators = [
    ('RandomForestClassifier', rf),
    ('KNeighborsClassifier', knn),
    ('MLPClassifier', mlpc),
    ('RidgeClassifier', rg),
]

# create the ensemble model
ensemble = VotingClassifier(estimators, voting='hard')
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
