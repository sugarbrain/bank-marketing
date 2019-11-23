# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Tira limite de vizualição do dataframe quando printado 
pandas.set_option('display.max_columns', None) 
pandas.set_option('display.max_rows', None)

seed = 42
np.random.seed(seed)

# Full train set
train_file = "../datasets/train.csv"
dataset = pandas.read_csv(train_file)

# use 20% of the train to search best params
train, _ = train_test_split(dataset, test_size=0.80, random_state=seed)

# KNN Params
metrics = ["manhattan", "euclidean", "chebyshev", "minkowski"]
n_neighbors = [x for x in range(3, 50) if x % 2 != 0]

params = []

for metric in metrics:
    for i, n in enumerate(n_neighbors):
        params.append({
            "id": metric[0:3].upper() + str(n),
            "metric": metric,
            "n_neighbors": n
        })

K_SPLITS = 2
X = train.values[:, 0:-1]
Y = train['y']

kf = StratifiedKFold(n_splits=K_SPLITS, random_state=seed)
kf.get_n_splits(X)

all_scores = []

print("Busca de Parametros KNN")
for param in params:
    clf = neighbors.KNeighborsClassifier(
        metric=param["metric"],
        n_neighbors=param["n_neighbors"]
    )

    scores = cross_val_score(clf, X, Y, cv=kf)

    mean = scores.mean()

    all_scores.append(mean)
    print("%s | %0.3f" % (param["id"], mean))

print(all_scores)

names = list(map(lambda x: x["id"], params))

plt.figure(figsize=(20, 8))

plt.plot(names, all_scores, 'o--')
plt.suptitle('Busca de Parametros KNN')
plt.show()

