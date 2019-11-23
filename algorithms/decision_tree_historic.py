# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import tree, preprocessing
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


# Decision Tree Params
def generate_dt_params():
    criterions = ["gini", "entropy"]
    max_depth = list(range(1, 50))

    params = []

    for criterion in criterions:
        for depth in max_depth:
            params.append({
                "id": criterion[0:3].upper() + str(depth),
                "criterion": criterion,
                "max_depth": depth
            })

    return params


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_dt_score(X, Y, params, kfold):
    print("Busca de Parametros Decision Tree")

    all_scores = []

    for param in params:
        clf = tree.DecisionTreeClassifier(criterion=param["criterion"],
                                          max_depth=param["max_depth"],
                                          random_state=SEED)

        scores = cross_val_score(clf, X, Y, cv=kfold)

        mean = scores.mean()

        all_scores.append({
            "id": param["id"],
            "criterion": param["criterion"],
            "max_depth": param["max_depth"],
            "result": mean
        })

        print("%s | %0.4f" % (param["id"], mean))

    best = max(all_scores, key=lambda s: s["result"])
    print(f"Best param: {best}")
    print(all_scores)
    return all_scores


def plot(scores):
    plt.figure(figsize=(50, 8))
    plt.margins(x=0.005)
    x = list(map(lambda x: x["id"], scores))  # names
    y = list(map(lambda x: x["result"], scores))  # scores

    plt.plot(x, y, 'o--')
    plt.suptitle('Busca de Parametros Decision Tree')
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()


def print_markdown_table(scores):
    print("Variação | *criterion* | *max_depth* | Acurácia média")
    print("------ | ------- | -------- | ----------")

    for s in scores:
        name = s["id"]
        criterion = s["criterion"]
        max_depth = s["max_depth"]
        result = '{:0.4f}'.format(s["result"])

        print(f"{name} | {criterion} | {max_depth} | {result}")


K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Generate params
params = generate_dt_params()

# Run scoring for best params
scores = run_dt_score(X, Y, params, kfold)

# plot
plot(scores)

print_markdown_table(scores)