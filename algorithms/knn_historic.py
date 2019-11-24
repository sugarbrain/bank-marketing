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

SEED = 42
np.random.seed(SEED)

# Full train set
train_file = "../datasets/train.csv"


def get_train_set(filepath, size=0.20):
    dataset = pandas.read_csv(train_file)

    test_size = 1.0 - size

    # use 20% of the train to search best params
    train, _ = train_test_split(dataset,
                                test_size=test_size,
                                random_state=SEED)

    return train


# KNN Params
def generate_knn_params():
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

    return params


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_knn_score(X, Y, params, kfold):
    print("Busca de Parametros KNN")

    all_scores = []

    for param in params:
        clf = neighbors.KNeighborsClassifier(metric=param["metric"],
                                             n_neighbors=param["n_neighbors"])

        scores = cross_val_score(clf, X, Y, cv=kfold)

        mean = scores.mean()

        all_scores.append({
            "id": param["id"],
            "metric": param["metric"],
            "n_neighbors": param["n_neighbors"],
            "result": mean
        })

        print("%s | %0.4f" % (param["id"], mean))

    best = max(all_scores, key=lambda s: s["result"])
    print(f"Best param: {best}")
    print(all_scores)
    return all_scores


def plot(scores):
    # options
    plt.figure(figsize=(25, 8))
    plt.margins(x=0.005)
    plt.rc('font', size=14)
    plt.xticks(rotation=90)
    plt.grid(linestyle='--')
    
    x = list(map(lambda x: x["id"], scores))  # names
    y = list(map(lambda x: x["result"], scores))  # scores

    plt.suptitle('Busca de Parametros KNN')
    plt.plot(x, y, 'o--')
    plt.show()


def print_markdown_table(scores):
    print("Variação | *metric* | *n_neighbors* | Acurácia média")
    print("------ | ------- | -------- | ----------")

    for s in scores:
        name = s["id"]
        metric = s["metric"]
        n = s["n_neighbors"]
        result = '{:0.4f}'.format(s["result"])

        print(f"{name} | {metric} | {n} | {result}")


K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Generate params
params = generate_knn_params()

# Run scoring for best params
scores = run_knn_score(X, Y, params, kfold)

# plot
plot(scores)

print_markdown_table(scores)