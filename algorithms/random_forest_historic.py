# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


# Random Forest Params
def generate_rf_params():
    n_estimators = list(range(1, 33))
    max_features = ["sqrt", "log2", None]

    params = []

    for feat in max_features:
        for num_estimators in n_estimators:
            features = str(feat)[0:1].upper()
            params.append({
                "id": f"{features}{num_estimators}",
                "n_estimators": num_estimators,
                "max_features": feat
            })

    return params


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_rf_score(X, Y, params, kfold):
    print("Busca de Parametros Random Forest")

    all_scores = []

    for param in params:
        clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                     max_features=param['max_features'],
                                     random_state=42)

        scores = cross_val_score(clf, X, Y, cv=kfold)

        mean = scores.mean()

        all_scores.append({
            "id": param["id"],
            "n_estimators": param["n_estimators"],
            "max_features": param["max_features"],
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
    plt.suptitle('Busca de Parametros Random Forest')
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()


def print_markdown_table(scores):
    print("Variação | *n_estimators* | *max_features* | Acurácia média")
    print("------ | ------- | -------- | ----------")

    for s in scores:
        name = s["id"]
        n_estimators = s["n_estimators"]
        max_features = s["max_features"]
        result = '{:0.4f}'.format(s["result"])

        print(f"{name} | {n_estimators} | {max_features} | {result}")


K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Generate params
params = generate_rf_params()

# Run scoring for best params
scores = run_rf_score(X, Y, params, kfold)

# plot
plot(scores)

print_markdown_table(scores)