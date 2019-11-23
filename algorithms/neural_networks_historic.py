# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


# Neural Networks Params
def generate_nn_params():
    solvers = ["lbfgs", "sgd", "adam"]
    # hidden_layer_sizes = (neurons_per_layer, number_of_layers)
    neurons_per_layer = list(range(1, 51))
    number_of_layers = list(range(1, 3))

    params = []

    for solver in solvers:
        for num_layers in number_of_layers:
            for num_neurons in neurons_per_layer:
                params.append({
                    "id": f"{solver[0:3].upper()}_{num_neurons}_{num_layers}",
                    "solver": solver,
                    "hidden_layer_sizes": (num_neurons, num_layers)
                })

    return params


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_nn_score(X, Y, params, kfold):
    print("Busca de Parametros Neural Networks")

    all_scores = []

    for param in params:
        clf = MLPClassifier(solver=param["solver"],
                            hidden_layer_sizes=param["hidden_layer_sizes"],
                            random_state=SEED)

        scores = cross_val_score(clf, X, Y, cv=kfold)

        mean = scores.mean()

        all_scores.append({
            "id": param["id"],
            "solver": param["solver"],
            "hidden_layer_sizes": param["hidden_layer_sizes"],
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
    plt.suptitle('Busca de Parametros Neural Networks')
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()


def print_markdown_table(scores):
    print("Variação | *solver* | *hidden_layer_sizes* | Acurácia média")
    print("------ | ------- | -------- | ----------")

    for s in scores:
        name = s["id"]
        solver = s["solver"]
        sizes = s["hidden_layer_sizes"]
        result = '{:0.4f}'.format(s["result"])

        print(f"{name} | {solver} | {sizes} | {result}")


K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Generate params
params = generate_nn_params()

# Run scoring for best params
scores = run_nn_score(X, Y, params, kfold)

# plot
plot(scores)

print_markdown_table(scores)